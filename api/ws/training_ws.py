"""WebSocket endpoint for streaming training progress."""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.services.pipeline_manager import get_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])


@router.websocket("/ws/training/{run_id}")
async def training_websocket(websocket: WebSocket, run_id: str):
    """Stream training progress events to the client.

    Sends buffered history first (for late-joining), then live events.
    """
    await websocket.accept()

    manager = get_manager()
    state = manager.get_run(run_id)

    if state is None:
        await websocket.send_json({"type": "error", "data": {"message": f"Run {run_id} not found"}})
        await websocket.close()
        return

    try:
        # Send buffered history
        for msg in state.history:
            await websocket.send_json(msg)

        # If already complete, close
        if state.status in ("trained", "complete", "error"):
            await websocket.close()
            return

        # Stream live events from the queue
        while True:
            try:
                msg = await asyncio.wait_for(state.progress_queue.get(), timeout=30.0)
                await websocket.send_json(msg)
                if msg.get("type") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping", "data": {}})

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from run {run_id}")
    except Exception as e:
        logger.exception(f"WebSocket error for run {run_id}")
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass
