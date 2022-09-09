import bentoml
from bentoml.io import JSON
from bentoml._internal.configuration.containers import BentoMLContainer

class EchoRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        # load the model instance
        self.tracer = BentoMLContainer.tracer_provider.get().get_tracer(__name__)

    @bentoml.Runnable.method(batchable=False)
    def echo(self, input_data: list[int]) -> list[int]:
        with self.tracer.start_as_current_span("custom_span"):
            import time
            time.sleep(0.1)
            return input_data

echo_runner = bentoml.Runner(EchoRunnable, name="echo_runner")
svc = bentoml.Service("echo_service", runners=[echo_runner])

@svc.api(input=JSON(), output=JSON())
def analysis(input_text: list[int]) -> list[int]:
    return echo_runner.echo.run(input_text)