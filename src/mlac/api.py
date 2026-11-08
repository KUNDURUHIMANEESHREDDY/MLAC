"""Optional FastAPI interface. State is process-local and non-durable."""
from dataclasses import asdict
from typing import Any
from mlac.evolution import EvolutionController
from mlac.ir import Architecture
from mlac.master_intelligence import MasterIntelligenceEngine
from mlac.registry import ModelRegistry
registry=ModelRegistry()
evolution=EvolutionController()
intelligence=MasterIntelligenceEngine()
def create_app() -> Any:
    try:
        from fastapi import FastAPI,HTTPException
    except ImportError as exc:
        raise RuntimeError("Install API dependencies: pip install 'mlac[api]'") from exc
    app=FastAPI(title="MLAC",version="0.7.0")
    @app.get("/health")
    def health(): return {"status":"ok","service":"mlac"}
    @app.post("/models")
    def register_model(payload:dict[str,Any]):
        try:
            record=registry.register(Architecture.from_dict(payload))
            return {"model_id":record.model_id,"architecture":record.architecture.to_dict()}
        except (KeyError,TypeError,ValueError) as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    @app.get("/models")
    def list_models():
        return [{"model_id":r.model_id,"architecture":r.architecture.to_dict(),"status":r.status,"created_at":r.created_at} for r in registry.list()]
    @app.get("/models/{model_id}")
    def get_model(model_id:str):
        try: r=registry.get(model_id)
        except KeyError as exc: raise HTTPException(status_code=404,detail=str(exc)) from exc
        return {"model_id":model_id,"architecture":r.architecture.to_dict()}
    @app.post("/models/{model_id}/proposals")
    def propose(model_id:str,payload:dict[str,Any]):
        try:
            record=registry.get(model_id)
            proposal=evolution.propose(model_id,record.architecture,
                hypothesis=str(payload.get("hypothesis","")),operation=str(payload.get("operation","")),
                parameters=payload.get("parameters",{}))
            return asdict(proposal)
        except KeyError as exc: raise HTTPException(status_code=404,detail=str(exc)) from exc
        except (ValueError,PermissionError) as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
    @app.post("/intelligence/plan")
    def intelligence_plan(payload:dict[str,Any]):
        try:
            report=intelligence.plan(model_id=str(payload.get("model_id","")),
                architecture=payload.get("architecture",{}),
                objective=str(payload.get("objective","")),
                constraints=payload.get("constraints"))
            return report.to_dict()
        except (KeyError,TypeError,ValueError) as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc
    @app.get("/proposals")
    def list_proposals(): return evolution.list_proposals()
    return app
app=create_app()
