# Notes from CG
Hey Jumal here are some notes if you're trying to run this for yourself.

## Codewiki
### Python & install
I ran into issues with this but I think that was my user error

### Running codewiki
This is the config that was provided in the docs
codewiki config set \
  --api-key YOUR_API_KEY \
  --base-url https://api.anthropic.com \
  --main-model claude-sonnet-4 \
  --cluster-model claude-sonnet-4 \
  --fallback-model glm-4p5

--base-url for anthropic is https://api.anthropic.com/v1
the models also need to be formatted like claude-sonnet-4-5-20250909
that information is in the claude api documentation.

The other issue is that the fallback-model couldn't be changed from glm-4p5.
Lastly the codewiki validate command always ended with the error "API connectivity failure". However the generate command still ran. 

The document output is missing two modules due to rate-limit issue and fallback-model mismatch.

### Sample error message
This first error message points to the problem that I was hitting the rate limit for claude
```
File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/_agent_graph.py", line 497, in _make_request
  |     model_response = await ctx.deps.model.request(message_history, model_settings, model_request_parameters)
  |                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/fallback.py", line 94, in request
  |     raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)
  | pydantic_ai.exceptions.FallbackExceptionGroup: All models from FallbackModel failed (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/openai.py", line 646, in _completions_create
    |     return await self.client.chat.completions.create(
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     ...<28 lines>...
    |     )
    |     ^
    |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/openai/resources/chat/completions/completions.py", line 2678, in create
    |     return await self._post(
    |            ^^^^^^^^^^^^^^^^^
    |     ...<49 lines>...
    |     )
    |     ^
    |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/openai/_base_client.py", line 1797, in post
    |     return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/openai/_base_client.py", line 1597, in request
    |     raise self._make_status_error_from_response(err.response) from None
    | openai.RateLimitError: Error code: 429 - {'error': {'code': 'rate_limit_error', 'message': 'This request would exceed the rate limit for your organization (66a0f89a-412d-4859-a6ce-d4743cd9596a) of 30,000 input tokens per minute. For details, refer to: https://docs.claude.com/en/api/rate-limits. You can see the response headers for current usage. Please reduce the prompt length or the maximum tokens requested, or try again later. You may also contact sales at https://www.anthropic.com/contact-sales to discuss your options for a rate limit increase.', 'type': 'invalid_request_error', 'param': None}}
    |
    | The above exception was the direct cause of the following exception:
    |
    | Traceback (most recent call last):
    |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/fallback.py", line 84, in request
    |     response = await model.request(messages, model_settings, model_request_parameters)
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/openai.py", line 559, in request
    |     response = await self._completions_create(
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |         messages, False, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    ```
    This second error is I think the switch to the fallback model which was not anthropic
    [16:02:28] ERROR    Error processing module diagnostics: All models from FallbackModel failed (2 sub-exceptions)
    [16:02:28] ERROR    Traceback:   + Exception Group Traceback (most recent call last):
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/codewiki/src/be/agent_orchestrator.py", line 133, in process_module
      |     result = await agent.run(
      |              ^^^^^^^^^^^^^^^^
      |     ...<7 lines>...
      |     )
      |     ^
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/agent/abstract.py", line 258, in run
      |     async with self.iter(
      |                ~~~~~~~~~^
      |         user_prompt=user_prompt,
      |         ^^^^^^^^^^^^^^^^^^^^^^^^
      |     ...<11 lines>...
      |         builtin_tools=builtin_tools,
      |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      |     ) as agent_run:
      |     ^
      |   File "/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/lib/python3.14/contextlib.py", line 235, in __aexit__
      |     await self.gen.athrow(value)
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/agent/__init__.py", line 705, in iter
      |     async with graph.iter(
      |                ~~~~~~~~~~^
      |         inputs=user_prompt_node,
      |         ^^^^^^^^^^^^^^^^^^^^^^^^
      |     ...<3 lines>...
      |         infer_name=False,
      |         ^^^^^^^^^^^^^^^^^
      |     ) as graph_run:
      |     ^
      |   File "/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/lib/python3.14/contextlib.py", line 235, in __aexit__
      |     await self.gen.athrow(value)
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_graph/beta/graph.py", line 270, in iter
      |     async with GraphRun[StateT, DepsT, OutputT](
      |                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
      |         graph=self,
      |         ^^^^^^^^^^^
      |     ...<3 lines>...
      |         traceparent=traceparent,
      |         ^^^^^^^^^^^^^^^^^^^^^^^^
      |     ) as graph_run:
      |     ^
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_graph/beta/graph.py", line 423, in __aexit__
      |     await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
      |   File "/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/lib/python3.14/contextlib.py", line 768, in __aexit__
      |     raise exc
      |   File "/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/lib/python3.14/contextlib.py", line 749, in __aexit__
      |     cb_suppress = cb(*exc_details)
      |   File "/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/lib/python3.14/contextlib.py", line 162, in __exit__
      |     self.gen.throw(value)
      |     ~~~~~~~~~~~~~~^^^^^^^
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_graph/beta/graph.py", line 981, in _unwrap_exception_groups
      |     raise exception
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_graph/beta/graph.py", line 750, in _run_tracked_task
      |     result = await self._run_task(t_)
      |              ^^^^^^^^^^^^^^^^^^^^^^^^
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_graph/beta/graph.py", line 782, in _run_task
      |     output = await node.call(step_context)
      |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_graph/beta/step.py", line 253, in _call_node
      |     return await node.run(GraphRunContext(state=ctx.state, deps=ctx.deps))
      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/_agent_graph.py", line 452, in run
      |     return await self._make_request(ctx)
      |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/_agent_graph.py", line 497, in _make_request
      |     model_response = await ctx.deps.model.request(message_history, model_settings, model_request_parameters)
      |                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/fallback.py", line 94, in request
      |     raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)
      | pydantic_ai.exceptions.FallbackExceptionGroup: All models from FallbackModel failed (2 sub-exceptions)
      +-+---------------- 1 ----------------
        | Traceback (most recent call last):
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/openai/_base_client.py", line 1532, in request
        |     response = await self._client.send(
        |                ^^^^^^^^^^^^^^^^^^^^^^^^
        |     ...<3 lines>...
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpx/_client.py", line 1629, in send
        |     response = await self._send_handling_auth(
        |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |     ...<4 lines>...
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpx/_client.py", line 1657, in _send_handling_auth
        |     response = await self._send_handling_redirects(
        |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |     ...<3 lines>...
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpx/_client.py", line 1694, in _send_handling_redirects
        |     response = await self._send_single_request(request)
        |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpx/_client.py", line 1730, in _send_single_request
        |     response = await transport.handle_async_request(request)
        |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpx/_transports/default.py", line 394, in handle_async_request
        |     resp = await self._pool.handle_async_request(req)
        |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpcore/_async/connection_pool.py", line 256, in handle_async_request
        |     raise exc from None
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpcore/_async/connection_pool.py", line 236, in handle_async_request
        |     response = await connection.handle_async_request(
        |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |         pool_request.request
        |         ^^^^^^^^^^^^^^^^^^^^
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpcore/_async/connection.py", line 103, in handle_async_request
        |     return await self._connection.handle_async_request(request)
        |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpcore/_async/http11.py", line 136, in handle_async_request
        |     raise exc
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpcore/_async/http11.py", line 106, in handle_async_request
        |     ) = await self._receive_response_headers(**kwargs)
        |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpcore/_async/http11.py", line 177, in _receive_response_headers
        |     event = await self._receive_event(timeout=timeout)
        |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpcore/_async/http11.py", line 217, in _receive_event
        |     data = await self._network_stream.read(
        |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |         self.READ_NUM_BYTES, timeout=timeout
        |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/httpcore/_backends/anyio.py", line 35, in read
        |     return await self._stream.receive(max_bytes=max_bytes)
        |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/anyio/streams/tls.py", line 244, in receive
        |     data = await self._call_sslobject_method(self._ssl_object.read, max_bytes)
        |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/anyio/streams/tls.py", line 180, in _call_sslobject_method
        |     result = func(*args)
        |   File "/opt/homebrew/Cellar/python@3.14/3.14.2/Frameworks/Python.framework/Versions/3.14/lib/python3.14/ssl.py", line 868, in read
        |     v = self._sslobj.read(len)
        | ssl.SSLError: [SSL: SSLV3_ALERT_BAD_RECORD_MAC] ssl/tls alert bad record mac (_ssl.c:2710)
        |
        | The above exception was the direct cause of the following exception:
        |
        | Traceback (most recent call last):
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/openai.py", line 646, in _completions_create
        |     return await self.client.chat.completions.create(
        |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |     ...<28 lines>...
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/openai/resources/chat/completions/completions.py", line 2678, in create
        |     return await self._post(
        |            ^^^^^^^^^^^^^^^^^
        |     ...<49 lines>...
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/openai/_base_client.py", line 1797, in post
        |     return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
        |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/openai/_base_client.py", line 1564, in request
        |     raise APIConnectionError(request=request) from err
        | openai.APIConnectionError: Connection error.
        |
        | The above exception was the direct cause of the following exception:
        |
        | Traceback (most recent call last):
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/fallback.py", line 84, in request
        |     response = await model.request(messages, model_settings, model_request_parameters)
        |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/openai.py", line 559, in request
        |     response = await self._completions_create(
        |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |         messages, False, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/openai.py", line 683, in _completions_create
        |     raise ModelAPIError(model_name=self.model_name, message=e.message) from e
        | pydantic_ai.exceptions.ModelAPIError: Connection error.
        +---------------- 2 ----------------
        | Traceback (most recent call last):
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/openai.py", line 646, in _completions_create
        |     return await self.client.chat.completions.create(
        |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |     ...<28 lines>...
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/openai/resources/chat/completions/completions.py", line 2678, in create
        |     return await self._post(
        |            ^^^^^^^^^^^^^^^^^
        |     ...<49 lines>...
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/openai/_base_client.py", line 1797, in post
        |     return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
        |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/openai/_base_client.py", line 1597, in request
        |     raise self._make_status_error_from_response(err.response) from None
        | openai.NotFoundError: Error code: 404 - {'error': {'code': 'not_found_error', 'message': 'model: glm-4p5', 'type': 'invalid_request_error', 'param': None}}
        |
        | The above exception was the direct cause of the following exception:
        |
        | Traceback (most recent call last):
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/fallback.py", line 84, in request
        |     response = await model.request(messages, model_settings, model_request_parameters)
        |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/openai.py", line 559, in request
        |     response = await self._completions_create(
        |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |         messages, False, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        |     )
        |     ^
        |   File "/Users/charlesgum/.local/pipx/venvs/codewiki/lib/python3.14/site-packages/pydantic_ai/models/openai.py", line 680, in _completions_create
        |     raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
        | pydantic_ai.exceptions.ModelHTTPError: status_code: 404, model_name: glm-4p5, body: {'code': 'not_found_error', 'message': 'model: glm-4p5', 'type': 'invalid_request_error', 'param': None}
        +------------------------------------
## Generated Document
Just as a note this knowledge folder was run on the muzero-general repo but I had originally stored it in a director called Project1-generate. That is why the overview mentions this.

### Thoughts on generated documents
I read though the overview and thought the quality was pretty good. I like that it actually included examples of code with an explaination provided.
