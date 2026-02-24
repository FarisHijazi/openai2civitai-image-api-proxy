"""
FastAPI proxy server to translate between OpenAI Images API and CivitAI API.

This application exposes OpenAI-compatible image generation endpoints and translates
requests to the CivitAI API backend, returning responses in OpenAI format.
"""

# TODO: add support for adding model URLs and that auto-converts to URNs

import traceback
import asyncio
import base64
import copy
import json
import os
from . import prompt_parser
import re
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Tuple
import sys

import backoff
import httpx
import httpcore
import joblib
from fastapi import FastAPI, Header, HTTPException, status, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from openai import APIConnectionError, APIError
from openai import APIResponse as APIResponse
from openai import (
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from openai import AsyncAPIResponse as AsyncAPIResponse
from openai import (
    AuthenticationError,
    BadRequestError,
    ConflictError,
    ContentFilterFinishReasonError,
    InternalServerError,
    LengthFinishReasonError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai.types import Image, ImagesResponse
from pydantic import BaseModel, Field, ValidationError


# import civitai_python.civitai as civitai
from .civitai_py import civitai
from . import prompt_reviser
from loguru import logger


"""
Configuration module for the OpenAI to CivitAI Image API Proxy.

This module handles environment variable loading and default settings.
"""


# Server Configuration
POLL_TIMEOUT: float = float(os.getenv("POLL_TIMEOUT", "600"))
POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "1"))

PROMPT_REVISOR_DISABLED_ERROR_MESSAGE = "<Prompt revisor is disabled, enable it by setting the following environment variables: PROMPT_REVISER_OPENAI_API_KEY=sk-asdfasdf ; PROMPT_REVISER_OPENAI_BASE_URL=https://api.openai.com/v1 ; PROMPT_REVISER_MODEL=gpt-4o-mini >"
DISABLE_NATIVE_BATCHING = os.getenv("DISABLE_NATIVE_BATCHING", "False").lower() in ["true", "1", "yes", "y"]


memory = joblib.Memory(location="./cache", verbose=0)


# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  # Keep stderr for important messages
logger.add(
    __file__ + ".log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
)


def loggo(log_level="debug"):
    def decorator(wrapped):
        def wrapper(*args, **kwargs):
            log_method = getattr(logger, log_level, logger.debug)
            log_method(f"Calling {wrapped.__name__} with args={args} kwargs={kwargs}")
            result = wrapped(*args, **kwargs)
            log_method(f"{wrapped.__name__} returned {result}")
            return result

        return wrapper

    return decorator

logger.debug(os.environ)

IMAGE_PROMPT_TEMPLATE = os.getenv("IMAGE_PROMPT_TEMPLATE", "{prompt}") or "{prompt}"
assert (
    IMAGE_PROMPT_TEMPLATE.count("{prompt}") == 1
), "IMAGE_PROMPT_TEMPLATE must contain exactly one {prompt}"

DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE = """### Task:
Generate a concise, 3-5 word title with an emoji summarizing the content in the content's primary language.
### Guidelines:
- The title should clearly represent the main theme or subject of the content.
- Use emojis that enhance understanding of the topic, but avoid quotation marks or special formatting.
- Write the title in the content's primary language.
- Prioritize accuracy over excessive creativity; keep it clear and simple.
- Your entire response must consist solely of the JSON object, without any introductory or concluding text.
- The output must be a single, raw JSON object, without any markdown code fences or other encapsulating text.
- Ensure no conversational text, affirmations, or explanations precede or follow the raw JSON output, as this will cause direct parsing failure.
### Output:
JSON format: { "title": "your concise title here" }
### Examples:
- { "title": "📉 Stock Market Trends" },
- { "title": "🍪 Perfect Chocolate Chip Recipe" },
- { "title": "Evolution of Music Streaming" },
- { "title": "Remote Work Productivity Tips" },
- { "title": "Artificial Intelligence in Healthcare" },
- { "title": "🎮 Video Game Development Insights" }
### Content:
<content>
${content}
</content>"""


class OpenAIImageRequest(BaseModel):
    """OpenAI-compatible image generation request model."""

    prompt: str = Field(
        ..., description="Text description of the desired image", max_length=4000
    )
    model: Optional[str] = Field(
        default=prompt_parser.CIVITAI_DEFAULT_MODEL,
        description="Model to use for image generation",
    )
    n: Optional[int] = Field(
        default=1, ge=1, le=100, description="Number of images to generate"
    )
    quality: Optional[Literal["standard", "hd"]] = Field(
        default=None, description="Image quality"
    )
    response_format: Optional[Literal["url", "b64_json"]] = Field(
        default="b64_json", description="Response format"
    )
    # don't forget to set it to "auto" to control the size through the prompt
    size: Optional[
        Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792", "auto"]
    ] = Field(default="auto", description="Size of generated images")
    style: Optional[Literal["vivid", "natural"]] = Field(
        default=None, description="Image style"
    )
    user: Optional[str] = Field(
        default=None, description="Unique identifier for the user"
    )
    seed: Optional[int] = Field(default=None, description="Seed for image generation")


class OpenAIChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    try:
        if prompt_reviser.is_available():
            logger.info("Prompt reviser is available")
        else:
            logger.info(PROMPT_REVISOR_DISABLED_ERROR_MESSAGE)
        assert os.environ[
            "CIVITAI_API_TOKEN"
        ], "CIVITAI_API_TOKEN environment variable is required. Please set it to your CivitAI API token."
        logger.info(f"✅ OpenAI to CivitAI Image API Proxy started successfully")
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 OpenAI to CivitAI Image API Proxy shutting down")


app = FastAPI(
    title="OpenAI to CivitAI Image API Proxy",
    description="A proxy server that translates between OpenAI Images API and CivitAI API",
    version="1.0.0",
    lifespan=lifespan,
)


# @app.middleware("http")
async def verbose_logging_middleware(request: Request, call_next):

    async def set_body(request: Request, body: bytes):
        async def receive():
            return {"type": "http.request", "body": body}

        request._receive = receive

    async def get_body(request: Request) -> bytes:
        body = await request.body()
        await set_body(request, body)
        return body

    request_id = os.urandom(8).hex()
    logger.info(f"Request ID: {request_id} - START")
    logger.info(f"Request ID: {request_id} - {request.method} {request.url}")
    logger.info(f"Request ID: {request_id} - Headers: {dict(request.headers)}")
    logger.info(
        f"Request ID: {request_id} - Client: {request.client.host}:{request.client.port}"
    )

    request_body = await get_body(request)
    if request_body:
        try:
            body_json = json.loads(request_body)
            logger.info(
                f"Request ID: {request_id} - Body: {json.dumps(body_json, indent=2)}"
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.info(
                f"Request ID: {request_id} - Body: {request_body.decode(errors='ignore')}"
            )

    start_time = time.time()

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(
            f"Request ID: {request_id} - Exception during request processing: {e}",
            exc_info=True,
        )
        raise e

    process_time = (time.time() - start_time) * 1000
    logger.info(f"Request ID: {request_id} - Processed in {process_time:.2f}ms")
    logger.info(f"Request ID: {request_id} - Response Status: {response.status_code}")
    logger.info(
        f"Request ID: {request_id} - Response Headers: {dict(response.headers)}"
    )

    if isinstance(response, StreamingResponse):
        logger.info(f"Request ID: {request_id} - Response Body: <StreamingResponse>")
    else:
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        if response_body:
            try:
                body_json = json.loads(response_body)

                def filter_urls_recursive(obj):
                    """Recursively replace URL-related values with '...' while keeping key names."""
                    if isinstance(obj, dict):
                        return {
                            k: (
                                "..."
                                if k in ["blobUrl", "blobUrls", "url"]
                                else filter_urls_recursive(v)
                            )
                            for k, v in obj.items()
                        }
                    elif isinstance(obj, list):
                        return [filter_urls_recursive(item) for item in obj]
                    else:
                        return obj

                filtered_body = filter_urls_recursive(body_json)
                logger.info(
                    f"Request ID: {request_id} - Response Body: {json.dumps(filtered_body, indent=2)}"
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.info(
                    f"Request ID: {request_id} - Response Body: {response_body.decode(errors='ignore')}"
                )

        response = Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    logger.info(f"Request ID: {request_id} - END")
    return response


@app.get("/v1/models")
async def list_models() -> dict:
    """List available models in OpenAI-compatible format so Open-WebUI can discover them."""
    default_model = prompt_parser.CIVITAI_DEFAULT_MODEL
    return {
        "object": "list",
        "data": [
            {
                "id": default_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "civitai",
            },
            {
                "id": "dall-e-3",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "civitai",
            },
        ],
    }


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "openai-civitai-proxy",
        "prompt_reviser available": prompt_reviser.is_available(),
        "DISABLE_NATIVE_BATCHING": DISABLE_NATIVE_BATCHING,
    }


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Request validation failed",
                "type": "invalid_request_error",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions in OpenAI-compatible format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": (
                    "api_error" if exc.status_code >= 500 else "invalid_request_error"
                ),
                "code": None,
            }
        },
    )


async def fetch_image_as_base64(url: str) -> str:
    """
    Fetches an image from a URL and returns it as a base64 encoded string.

    Args:
        url: The URL of the image to fetch.

    Returns:
        The base64 encoded image content.

    Raises:
        HTTPException: If the image cannot be fetched.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error fetching image: {e.response.status_code} from {url}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch image from provider: {e.response.status_code}",
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching image: {e}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error fetching image: {str(e)}"
        )


@loggo("trace")
def translate_openai_to_civitai(request: OpenAIImageRequest, i: int = 0) -> dict:
    """
    Translate OpenAI image request format to CivitAI format.

    Args:
        request: OpenAI-formatted image generation request
        i: Index for batching, used to vary seeds.

    Returns:
        Dictionary formatted for CivitAI API
    """

    original_is_gen_data = prompt_parser.is_generation_data_format(request.prompt)

    if not original_is_gen_data:
        logger.info(
            f"Prompt is not in generation data format, adding image prompt template"
        )
        request.prompt = (
            IMAGE_PROMPT_TEMPLATE.replace("\\n", "\n")
            .replace("\\", "")
            .strip()
            .format(
                prompt=request.prompt.replace("\\n", "\n").replace("\\", "").strip()
            )
            .strip()
        )

    generation_input = prompt_parser.parse_inputstring_to_generation_input(
        request.prompt
    )

    if request.seed:
        generation_input["params"]["seed"] = request.seed

    generation_input["quantity"] = request.n

    if request.size and request.size != "auto":
        width, height = request.size.split("x")
        generation_input["params"]["width"] = int(width)
        generation_input["params"]["height"] = int(height)

    if "seed" in generation_input["params"]:
        generation_input["params"]["seed"] += i

    # priority 1: generation data format
    if original_is_gen_data:
        logger.info(f"Prompt is in generation data format, no revision")
        generation_input["params"]["prompt"] = generation_input["params"][
            "prompt"
        ].strip()
        return generation_input

    if original_is_gen_data:
        generation_input["params"]["revised_prompt"] = PROMPT_REVISOR_DISABLED_ERROR_MESSAGE
        return generation_input

    if prompt_reviser.is_available():
        # it's ok to mess with the prompt now that we're revising it anyway, RNG is already destroyed
        generation_input["params"]["prompt"] = (
            generation_input["params"]["prompt"].replace("\n", " ").replace("  ", " ")
            + " " * i
        )
        # WARNING: this function takes a long time
        revised_prompt = (
            prompt_reviser.revise_prompt(generation_input["params"]["prompt"])
            .replace("\n", " ")
            .replace("  ", " ")
            .strip()
        )

        generation_input["params"]["revised_prompt"] = revised_prompt
        logger.info(
            f"Revised prompt from:\"{generation_input['params']['prompt']}\" -> to:\"{generation_input['params']['revised_prompt']}\""
        )
        generation_input["params"]["prompt"] = generation_input["params"]["revised_prompt"]

    # Map quality to CivitAI parameters
    if request.quality:
        # Higher quality = more steps and lower CFG scale for better results
        if request.quality == "hd":
            generation_input["params"]["cfgScale"] = 4.0
            generation_input["params"]["steps"] = 30
        else:  # "standard"
            generation_input["params"]["cfgScale"] = 7.0
            generation_input["params"]["steps"] = 20

    # Use the configured default CivitAI model
    if request.style:
        # Adjust negative prompt based on style
        if request.style == "natural":
            generation_input["params"][
                "prompt"
            ] += ", natural lighting, soft colors, realistic style, subtle details"
        else:  # "vivid"
            generation_input["params"][
                "prompt"
            ] += ", vibrant colors, high contrast, dramatic lighting, bold details, saturated"

    return generation_input


@loggo("trace")
async def translate_civitai_to_openai(
    civitai_response: dict, request: OpenAIImageRequest, civitai_input: dict
) -> ImagesResponse:
    """
    Translate CivitAI response format to OpenAI format.

    Args:
        civitai_response: Response from CivitAI API
        request: Original OpenAI request for context

    Returns:
        OpenAI-formatted response

    Raises:
        ValueError: If response format is invalid
    """
    try:

        results = [
            result for job in civitai_response["jobs"] for result in job["result"] if result.get("available")
        ]
        if not results:
            raise ValueError("No results found in CivitAI response")

        # Get the image URL
        assert all([result.get("blobUrl") for result in results]), "No image URL found in result"

        image_urls = [result.get("blobUrl") for result in results]

        image_data_list = []
        for image_url in image_urls:
            # Create image data based on requested format
            b64_json_for_response = None

            if request.response_format == "url":
                image_url_for_response = image_url
            elif request.response_format == "b64_json":
                b64_json_for_response = await fetch_image_as_base64(image_url)
                image_url_for_response = (
                    "data:image/png;base64," + b64_json_for_response
                )

            if civitai_input["params"].get("revised_prompt"):
                prompt_revision_message = civitai_input["params"]["revised_prompt"]
            elif not prompt_reviser.is_available():
                prompt_revision_message = PROMPT_REVISOR_DISABLED_ERROR_MESSAGE
            elif prompt_parser.is_generation_data_format(request.prompt):
                prompt_revision_message = f'<prompt not revised because it is in generation data format (with "Negative prompt:", "Additional networks:", etc)>'
            else:
                raise ValueError(
                    f"Prompt revision message not found for prompts: {request.prompt=}, {prompt_reviser.is_available()=}, {prompt_parser.is_generation_data_format(request.prompt)=}, {civitai_input['params']['revised_prompt']=}, {civitai_input['params']['prompt']=}"
                )

            image_data_list.append(Image(
                url=image_url_for_response,
                b64_json=b64_json_for_response,
                revised_prompt=prompt_revision_message,
            ))

        # Use job creation time if available, otherwise current time
        created_time = civitai_response["jobs"][0].get("createdAt", int(time.time()))
        if isinstance(created_time, str):
            # If timestamp is a string, try to parse it
            import datetime

            try:
                dt = datetime.datetime.fromisoformat(
                    created_time.replace("Z", "+00:00")
                )
                created_time = int(dt.timestamp())
            except:
                created_time = int(time.time())

        return ImagesResponse(created=created_time, data=image_data_list)

    except KeyError as e:
        raise ValueError(f"Missing required field in CivitAI response: {e}")
    except Exception as e:
        raise ValueError(f"Failed to parse CivitAI response: {e}")


@loggo("trace")
def civitai_jobs_get(token, job_id):
    return civitai.jobs.get(token, job_id)


@loggo("trace")
async def poll_civitai_job(token: str, job_id: str) -> dict:
    """
    Poll CivitAI job until completion.

    Args:
        token: Job token from CivitAI
        job_id: Job ID to poll

    Returns:
        Final job result

    Raises:
        HTTPException: If job fails or times out
    """
    logger.info(f"🔄 Starting to poll job {job_id} (max {POLL_TIMEOUT} seconds)")

    start_time = time.time()
    attempt = 0

    while (time.time() - start_time) < POLL_TIMEOUT:
        attempt += 1
        try:
            # wait=False, so the response looks something like this:
            #
            # {
            #     "token": response.token,
            #     "jobs": [{
            #         "jobId": job.jobId,
            #         "cost": job.cost,
            #         "result": job.result,
            #         "scheduled": job.scheduled,
            #     } for job in response.jobs]
            # }
            response = await asyncio.to_thread(
                civitai_jobs_get, token=token, job_id=job_id
            )

            if "jobs" not in response or len(response["jobs"]) == 0:
                logger.warning(f"⚠️ No jobs found in response on attempt {attempt}")
                await asyncio.sleep(POLL_INTERVAL)
                continue


            job_statuses = [job.get("status", "unknown") for job in response["jobs"]]

            elapsed_time = time.time() - start_time
            logger.info(
                f"📊 Attempt {attempt} ({elapsed_time:.1f}s elapsed): Job status = {job_statuses}"
            )

            # Check if job failed
            for job, job_status in zip(response["jobs"], job_statuses):
                if job_status in ["Failed", "Cancelled"]:
                    error_msg = job.get(
                        "error", "Job failed without specific error message"
                )
                    logger.error(f"❌ Job failed with status {job_status}: {error_msg}")
                    raise HTTPException(
                        status_code=500, detail=f"CivitAI job failed: {error_msg}"
                    )

            # Check if job is complete
            if (
                job.get("result")
                and len(job["result"]) > 0
                and all([result.get("available") for result in job["result"]])
            ):
                logger.info(
                    f"✅ Job completed successfully after {attempt} attempts ({elapsed_time:.1f}s)"
                )
                return response
            else:
                logger.info(
                    f"⏳ Job still processing (attempt {attempt}), time elapsed: {elapsed_time:.1f}s/{POLL_TIMEOUT}s"
                )

            # Wait before next poll
            await asyncio.sleep(POLL_INTERVAL)

        except HTTPException:
            # Re-raise HTTP exceptions (like job failures)
            raise
        except Exception as e:
            logger.error(f"⚠️ Error polling job {job_id} on attempt {attempt}: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    logger.error(f"⏰ Job {job_id} timed out after {POLL_TIMEOUT} seconds")
    raise HTTPException(
        status_code=408,
        detail=f"Job timed out after {POLL_TIMEOUT} seconds. Try again or use a simpler prompt.",
    )


# @memory.cache
@backoff.on_exception(
    backoff.expo, 
    (
        # httpx timeout exceptions
        httpx.TimeoutException,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        # httpcore timeout exceptions (underlying layer)
        httpcore.TimeoutException,
        httpcore.ReadTimeout,
        httpcore.WriteTimeout,
        httpcore.ConnectTimeout,
        httpcore.PoolTimeout,
        # General exceptions
        Exception,
        HTTPException,
    ),
    max_tries=10, 
    jitter=backoff.full_jitter,
    on_backoff=lambda details: logger.warning(f"🔄 Retrying civitai_image_create after {details['exception'].__class__.__name__}: {details['exception']} (attempt {details['tries']}/{10})"),
)
@loggo("trace")
async def civitai_image_create(civitai_input: dict, wait: bool = False):
    # logger.info(f"🔄 Submitting CivitAI image creation request: {civitai_input=}")
    return await asyncio.to_thread(civitai.image.create, civitai_input, wait=wait)


@app.post("/v1/images/generations", response_model=ImagesResponse)
async def create_image(
    request: OpenAIImageRequest,
    authorization: Optional[str] = Header(
        None, description="Bearer token for authentication"
    ),
) -> ImagesResponse:
    """
    Create an image using OpenAI-compatible API that proxies to CivitAI.

    Args:
        request: Image generation request in OpenAI format
        authorization: Optional authorization header

    Returns:
        Generated image response in OpenAI format

    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Validate API token is configured
        if not os.environ["CIVITAI_API_TOKEN"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="CivitAI API token not configured",
            )

        # Validate prompt length
        if len(request.prompt.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt cannot be empty"
            )
        if "###Task:".lower() in request.prompt.lower().replace("\n", " ").replace(
            " ", ""
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be a title generation prompt",
            )

        logger.info(
            f"🎨 Submitting {request.n} image generation request(s): {request.size}, quality={request.quality}, style={request.style}"
        )

        loop = asyncio.get_running_loop()

        # Step 1: Translate OpenAI requests to Civitai inputs in parallel threads.
        # This is important because prompt revision can be slow.

        quantity = request.n or 1
        assert quantity > 0, "Quantity must be greater than 0"
        request.n = quantity
        if DISABLE_NATIVE_BATCHING:
            request.n = 1
            translate_tasks = [
                loop.run_in_executor(None, translate_openai_to_civitai, copy.deepcopy(request), i)
                for i in range(quantity)
            ]
            request.n = quantity
        else:
            translate_tasks = [loop.run_in_executor(None, translate_openai_to_civitai, copy.deepcopy(request), 0)]
        civitai_inputs = await asyncio.gather(*translate_tasks)

        # Step 2: Create image generation jobs on Civitai in parallel.
        create_tasks = [
            civitai_image_create(civitai_input, wait=False) for civitai_input in civitai_inputs
        ]
        civitai_responses = await asyncio.gather(*create_tasks)

        # Step 3: Poll for job completion in parallel.
        polling_tasks = []
        for res in civitai_responses:
            if res and res.get("token") and res.get("jobs"):
                polling_tasks += [
                    poll_civitai_job(res["token"], job["jobId"])
                    for job in res["jobs"]
                ]

        final_responses = await asyncio.gather(*polling_tasks)

        # Step 4: Translate Civitai responses to OpenAI format in parallel.
        translation_tasks = [
            translate_civitai_to_openai(final_res, request, civitai_input)
            for final_res, civitai_input in zip(final_responses, civitai_inputs)
        ]
        openai_responses = await asyncio.gather(*translation_tasks)

        # Combine all image data into single response
        all_image_data: List[Image] = [
            data for res in openai_responses for data in res.data
        ]


        return ImagesResponse(created=openai_responses[0].created, data=all_image_data)

    except HTTPException as e:
        logger.error(f"HTTPException: {str(e)}")
        raise
    except ValidationError as e:
        logger.error(f"ValidationError: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error generating image: {e} {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while generating the image: {str(e)}",
        )


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: OpenAIChatRequest,
    authorization: Optional[str] = Header(None),
):
    # TODO: make this respond with mocks
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header with Bearer token is required.",
        )

    openai_api_key = authorization.split(" ")[1]
    client = AsyncOpenAI(api_key=openai_api_key)

    async def stream_openai_response():
        try:
            stream = await client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            )
            async for chunk in stream:
                yield f"data: {json.dumps(chunk.model_dump(exclude_unset=True))}\n\n"
        except OpenAIError as e:
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "openai_error",
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(
            stream_openai_response(),
            media_type="text/event-stream",
        )
    else:
        try:
            response = await client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )
            return JSONResponse(content=response.model_dump())
        except OpenAIError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )


def main():
    """Entry point for running the server."""
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "openai2civitai.server:app",
        host=host,
        port=port,
        reload=False
    )


if __name__ == "__main__":
    main()
