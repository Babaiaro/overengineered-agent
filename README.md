# GreetingIntelligenceOrchestrationPlatform (GIOP)

An enterprise-grade, cloud-native\*, AI-powered\*\* agent that determines whether to say "hello".

\* not actually cloud-native
\*\* contains zero (0) machine learning

## Architecture

| Layer | Purpose | Why it's there |
|---|---|---|
| `IntentClassificationStrategy` | Decides if an utterance wants a greeting | Because an `if` statement needed an interface |
| `AbstractGreetingComponentFactory` | Produces the strategy | An abstract factory, for the one strategy that exists |
| `GreetingComponentFactoryProvider` | Thread-safe singleton that produces the factory | The factory needed a provider |
| `InversionOfControlContainer` / `ContainerBootstrapper` / `ContainerBootstrapperFactory` | Resolves the provider that provides the factory | Direct instantiation was too direct |
| `EventBus` + `LoggingTelemetryObserver` | Publishes lifecycle events | Pub/sub for three log lines |
| `Command` / `CommandFactory` | Turns a decision into a response | GoF Command pattern for `f"Hello, {name}!"` |
| `CommandInterceptorChain` (`LoggingCommandInterceptor`, `TimingCommandInterceptor`) | Wraps command execution in middleware | Measures the microseconds spent returning `""` |
| `UtteranceIngestionQueueWorker` | Background thread consuming a queue | Horizontally scales to one thread |

## Usage

```bash
python3 overengineered_agent.py
```

Feeds it `"hey there"`, `"quiet please"`, and `"asdkjaslkdj"`. Only the first produces output: `Hello, Bob!`

## Roadmap

- [ ] Kubernetes
- [ ] A second `if` statement
- [ ] Blockchain
