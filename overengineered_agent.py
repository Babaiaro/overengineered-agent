"""
GreetingIntelligenceOrchestrationPlatform (GIOP)
An enterprise-grade, cloud-native*, AI-powered** agent for saying "hello".

*  not actually cloud-native
** contains zero (0) machine learning
"""

from __future__ import annotations

import abc
import enum
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Domain layer: Value Objects
# ---------------------------------------------------------------------------

class IntentClassification(enum.Enum):
    GREETING_DESIRED = "GREETING_DESIRED"
    GREETING_UNDESIRED = "GREETING_UNDESIRED"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class Utterance:
    text: str
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    received_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class AgentDecision:
    classification: IntentClassification
    confidence: float
    correlation_id: str


# ---------------------------------------------------------------------------
# Strategy layer: Intent Classification Strategies
# ---------------------------------------------------------------------------

@runtime_checkable
class IntentClassificationStrategy(Protocol):
    def classify(self, utterance: Utterance) -> IntentClassification: ...


class KeywordPresenceIntentClassificationStrategy(IntentClassificationStrategy):
    """Classifies intent via a sophisticated lexical substring analysis."""

    _POSITIVE_LEXICON = frozenset({"hi", "hello", "hey", "yo"})
    _NEGATIVE_LEXICON = frozenset({"no", "quiet", "silence"})

    def classify(self, utterance: Utterance) -> IntentClassification:
        tokens = set(utterance.text.lower().split())
        if tokens & self._NEGATIVE_LEXICON:
            return IntentClassification.GREETING_UNDESIRED
        if tokens & self._POSITIVE_LEXICON:
            return IntentClassification.GREETING_DESIRED
        return IntentClassification.INDETERMINATE


class IntentClassificationStrategyFactory:
    """Factory for producing IntentClassificationStrategy instances."""

    @staticmethod
    def create_default_strategy() -> IntentClassificationStrategy:
        return KeywordPresenceIntentClassificationStrategy()


# ---------------------------------------------------------------------------
# Abstract Factory for the Abstract Factory (you read that right)
# ---------------------------------------------------------------------------

class AbstractGreetingComponentFactory(abc.ABC):
    @abc.abstractmethod
    def create_strategy(self) -> IntentClassificationStrategy: ...


class DefaultGreetingComponentFactory(AbstractGreetingComponentFactory):
    def create_strategy(self) -> IntentClassificationStrategy:
        return IntentClassificationStrategyFactory.create_default_strategy()


class GreetingComponentFactoryProvider:
    """Provides the factory that provides the strategy. Factories all the way down."""

    _instance: "GreetingComponentFactoryProvider | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def get_factory(self) -> AbstractGreetingComponentFactory:
        return DefaultGreetingComponentFactory()


# ---------------------------------------------------------------------------
# Inversion-of-Control layer: because instantiating a class directly
# would have been far too straightforward
# ---------------------------------------------------------------------------

class ServiceRegistrationKey(enum.Enum):
    GREETING_COMPONENT_FACTORY_PROVIDER = "GREETING_COMPONENT_FACTORY_PROVIDER"


class ServiceRegistration:
    def __init__(self, factory: Callable[[], Any], singleton: bool = True) -> None:
        self._factory = factory
        self._singleton = singleton
        self._instance: Any | None = None

    def resolve(self) -> Any:
        if not self._singleton:
            return self._factory()
        if self._instance is None:
            self._instance = self._factory()
        return self._instance


class InversionOfControlContainer:
    """A registry of registrations that resolve providers that provide factories."""

    def __init__(self) -> None:
        self._registrations: dict[ServiceRegistrationKey, ServiceRegistration] = {}

    def register(self, key: ServiceRegistrationKey, factory: Callable[[], Any]) -> None:
        self._registrations[key] = ServiceRegistration(factory)

    def resolve(self, key: ServiceRegistrationKey) -> Any:
        return self._registrations[key].resolve()


class ContainerBootstrapper:
    """Bootstraps the container that resolves the provider that provides the factory."""

    @staticmethod
    def bootstrap() -> InversionOfControlContainer:
        container = InversionOfControlContainer()
        container.register(
            ServiceRegistrationKey.GREETING_COMPONENT_FACTORY_PROVIDER,
            GreetingComponentFactoryProvider,
        )
        return container


class ContainerBootstrapperFactory:
    """Produces bootstrappers for containers. Someone had to."""

    @staticmethod
    def create() -> ContainerBootstrapper:
        return ContainerBootstrapper()


# ---------------------------------------------------------------------------
# Observer layer: Telemetry / Event Bus
# ---------------------------------------------------------------------------

class AgentEvent(enum.Enum):
    UTTERANCE_RECEIVED = "UTTERANCE_RECEIVED"
    DECISION_MADE = "DECISION_MADE"
    RESPONSE_EMITTED = "RESPONSE_EMITTED"


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[AgentEvent, list[Callable[[Any], None]]] = {}

    def subscribe(self, event: AgentEvent, handler: Callable[[Any], None]) -> None:
        self._subscribers.setdefault(event, []).append(handler)

    def publish(self, event: AgentEvent, payload: Any) -> None:
        for handler in self._subscribers.get(event, []):
            handler(payload)


class LoggingTelemetryObserver:
    """A dedicated microservice-shaped observer for structured observability."""

    def __init__(self, event_bus: EventBus) -> None:
        self._logger = logging.getLogger("GIOP.Telemetry")
        event_bus.subscribe(AgentEvent.UTTERANCE_RECEIVED, self._on_utterance)
        event_bus.subscribe(AgentEvent.DECISION_MADE, self._on_decision)
        event_bus.subscribe(AgentEvent.RESPONSE_EMITTED, self._on_response)

    def _on_utterance(self, payload: Utterance) -> None:
        self._logger.info("UTTERANCE_RECEIVED cid=%s text=%r", payload.correlation_id, payload.text)

    def _on_decision(self, payload: AgentDecision) -> None:
        self._logger.info(
            "DECISION_MADE cid=%s classification=%s confidence=%.2f",
            payload.correlation_id, payload.classification.value, payload.confidence,
        )

    def _on_response(self, payload: str) -> None:
        self._logger.info("RESPONSE_EMITTED text=%r", payload)


# ---------------------------------------------------------------------------
# Command layer
# ---------------------------------------------------------------------------

class Command(abc.ABC):
    @abc.abstractmethod
    def execute(self) -> Any: ...


class EmitGreetingCommand(Command):
    def __init__(self, recipient: str) -> None:
        self._recipient = recipient

    def execute(self) -> str:
        return f"Hello, {self._recipient}!"


class NoOpCommand(Command):
    def execute(self) -> str:
        return ""


class CommandFactory:
    @staticmethod
    def for_classification(classification: IntentClassification, recipient: str) -> Command:
        if classification == IntentClassification.GREETING_DESIRED:
            return EmitGreetingCommand(recipient)
        return NoOpCommand()


# ---------------------------------------------------------------------------
# Middleware layer: Command Interceptor Chain
# (a function call, mediated by a chain of objects that agree to call
#  the next object in the chain)
# ---------------------------------------------------------------------------

class CommandInterceptor(abc.ABC):
    @abc.abstractmethod
    def intercept(self, command: Command, proceed: Callable[[], Any]) -> Any: ...


class LoggingCommandInterceptor(CommandInterceptor):
    def __init__(self) -> None:
        self._logger = logging.getLogger("GIOP.Interceptor.Logging")

    def intercept(self, command: Command, proceed: Callable[[], Any]) -> Any:
        self._logger.info("Invoking command=%s", type(command).__name__)
        result = proceed()
        self._logger.info("Command=%s returned %r", type(command).__name__, result)
        return result


class TimingCommandInterceptor(CommandInterceptor):
    def __init__(self) -> None:
        self._logger = logging.getLogger("GIOP.Interceptor.Timing")

    def intercept(self, command: Command, proceed: Callable[[], Any]) -> Any:
        started_at = time.perf_counter()
        result = proceed()
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        self._logger.info("Command=%s took %.4fms", type(command).__name__, elapsed_ms)
        return result


class CommandInterceptorChain:
    """Wraps command.execute() in an arbitrarily deep onion of middleware."""

    def __init__(self, interceptors: list[CommandInterceptor]) -> None:
        self._interceptors = interceptors

    def invoke(self, command: Command) -> Any:
        def build(index: int) -> Callable[[], Any]:
            if index >= len(self._interceptors):
                return command.execute
            return lambda: self._interceptors[index].intercept(command, build(index + 1))

        return build(0)()


class CommandInterceptorChainFactory:
    @staticmethod
    def create_default_chain() -> CommandInterceptorChain:
        return CommandInterceptorChain([LoggingCommandInterceptor(), TimingCommandInterceptor()])


# ---------------------------------------------------------------------------
# Async work-queue layer (because a function call wasn't distributed enough)
# ---------------------------------------------------------------------------

class UtteranceIngestionQueueWorker(threading.Thread):
    def __init__(self, work_queue: "queue.Queue[Utterance]", on_item: Callable[[Utterance], None]) -> None:
        super().__init__(daemon=True)
        self._queue = work_queue
        self._on_item = on_item
        self._running = True

    def run(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._on_item(item)
            self._queue.task_done()

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# The Orchestration Platform itself
# ---------------------------------------------------------------------------

class GreetingIntelligenceOrchestrationPlatform:
    """
    A horizontally-scalable(*), fault-tolerant(**), AI-driven(***) agent
    that determines whether to say "hello".

    (*)   runs on one thread
    (**)  has no error handling
    (***) if-statements
    """

    def __init__(self, recipient: str = "World") -> None:
        self._recipient = recipient
        self._container = ContainerBootstrapperFactory.create().bootstrap()
        provider: GreetingComponentFactoryProvider = self._container.resolve(
            ServiceRegistrationKey.GREETING_COMPONENT_FACTORY_PROVIDER
        )
        self._factory = provider.get_factory()
        self._strategy = self._factory.create_strategy()
        self._event_bus = EventBus()
        self._telemetry = LoggingTelemetryObserver(self._event_bus)
        self._interceptor_chain = CommandInterceptorChainFactory.create_default_chain()
        self._ingestion_queue: "queue.Queue[Utterance]" = queue.Queue()
        self._results: list[str] = []
        self._worker = UtteranceIngestionQueueWorker(self._ingestion_queue, self._process)
        self._worker.start()

    def submit(self, text: str) -> None:
        utterance = Utterance(text=text)
        self._event_bus.publish(AgentEvent.UTTERANCE_RECEIVED, utterance)
        self._ingestion_queue.put(utterance)

    def _process(self, utterance: Utterance) -> None:
        classification = self._strategy.classify(utterance)
        decision = AgentDecision(
            classification=classification,
            confidence=1.0 if classification != IntentClassification.INDETERMINATE else 0.0,
            correlation_id=utterance.correlation_id,
        )
        self._event_bus.publish(AgentEvent.DECISION_MADE, decision)

        command = CommandFactory.for_classification(decision.classification, self._recipient)
        response = self._interceptor_chain.invoke(command)
        if response:
            self._event_bus.publish(AgentEvent.RESPONSE_EMITTED, response)
            self._results.append(response)

    def shutdown(self) -> list[str]:
        self._ingestion_queue.join()
        self._worker.stop()
        return self._results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    platform = GreetingIntelligenceOrchestrationPlatform(recipient="Bob")
    platform.submit("hey there")
    platform.submit("quiet please")
    platform.submit("asdkjaslkdj")
    results = platform.shutdown()
    print("\nFinal agent output:", results)


if __name__ == "__main__":
    main()
