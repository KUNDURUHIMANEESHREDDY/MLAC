# MLAC architecture

The MVP separates architecture metadata, inspection, candidate proposals, and governance.

- **IR:** versioned, framework-neutral component and tensor metadata.
- **Inspector:** reads an already-instantiated PyTorch module; does not load checkpoints or run a forward pass.
- **Registry:** in-memory model metadata store.
- **Mutation:** explicit dropout-configuration update on an IR snapshot. This is not a live graph or weight rewrite.
- **Evolution:** records hypotheses as proposals; it does not execute generated code, train candidates, or promote deployments.
- **Governance:** candidate changes can be proposed; production mutations are default-deny.

Install the API extra and run `mlac serve`. Keep the unauthenticated development API on localhost. Before network exposure, add authentication, authorization, durable storage, rate limits, artifact isolation, and an independently enforced sandbox.

PostgreSQL, object storage, MLflow, Temporal, distributed training, model-provider adapters, and hardware schedulers remain roadmap items.
