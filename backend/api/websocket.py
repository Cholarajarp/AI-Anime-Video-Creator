"""
WebSocket Handler for Real-time Updates
========================================
Provides real-time job progress and status updates via WebSocket.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional
import asyncio
import json
from loguru import logger
from dataclasses import dataclass


@dataclass
class ClientConnection:
    """Represents a connected WebSocket client."""
    websocket: WebSocket
    client_id: str
    subscribed_jobs: Set[str]


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts.

    Supports:
    - Multiple clients connected simultaneously
    - Per-job subscriptions
    - Broadcast to all clients
    - Targeted messages to specific clients
    """

    def __init__(self):
        self.active_connections: Dict[str, ClientConnection] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections[client_id] = ClientConnection(
                websocket=websocket,
                client_id=client_id,
                subscribed_jobs=set()
            )
        logger.info(f"Client {client_id} connected. Total: {len(self.active_connections)}")

    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
        logger.info(f"Client {client_id} disconnected. Total: {len(self.active_connections)}")

    async def subscribe_to_job(self, client_id: str, job_id: str):
        """Subscribe a client to updates for a specific job."""
        async with self._lock:
            if client_id in self.active_connections:
                self.active_connections[client_id].subscribed_jobs.add(job_id)

    async def unsubscribe_from_job(self, client_id: str, job_id: str):
        """Unsubscribe a client from a specific job."""
        async with self._lock:
            if client_id in self.active_connections:
                self.active_connections[client_id].subscribed_jobs.discard(job_id)

    async def send_personal(self, client_id: str, message: dict):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to client {client_id}: {e}")
                await self.disconnect(client_id)

    async def broadcast(self, message: dict):
        """Send a message to all connected clients."""
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to {client_id}: {e}")
                disconnected.append(client_id)

        for client_id in disconnected:
            await self.disconnect(client_id)

    async def broadcast_job_update(self, job_id: str, message: dict):
        """Send a job update to all clients subscribed to that job."""
        disconnected = []
        for client_id, connection in self.active_connections.items():
            # Send to clients subscribed to this job or with empty subscription (all jobs)
            if job_id in connection.subscribed_jobs or len(connection.subscribed_jobs) == 0:
                try:
                    await connection.websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send job update to {client_id}: {e}")
                    disconnected.append(client_id)

        for client_id in disconnected:
            await self.disconnect(client_id)

    async def send_progress_update(
        self,
        job_id: str,
        progress: float,
        message: str
    ):
        """Send a progress update for a job."""
        await self.broadcast_job_update(job_id, {
            "type": "progress",
            "job_id": job_id,
            "progress": progress,
            "message": message
        })

    async def send_status_update(self, job_id: str, status: str):
        """Send a status change for a job."""
        await self.broadcast_job_update(job_id, {
            "type": "status",
            "job_id": job_id,
            "status": status
        })

    async def send_job_complete(
        self,
        job_id: str,
        output_video: str,
        output_audio: str,
        thumbnail: Optional[str] = None
    ):
        """Send job completion notification."""
        await self.broadcast_job_update(job_id, {
            "type": "complete",
            "job_id": job_id,
            "output_video": output_video,
            "output_audio": output_audio,
            "thumbnail": thumbnail
        })

    async def send_job_error(self, job_id: str, error: str):
        """Send job error notification."""
        await self.broadcast_job_update(job_id, {
            "type": "error",
            "job_id": job_id,
            "error": error
        })

    async def send_queue_update(self, queue_stats: dict):
        """Send queue statistics update to all clients."""
        await self.broadcast({
            "type": "queue_update",
            **queue_stats
        })

    async def send_system_status(self, status: dict):
        """Send system status update to all clients."""
        await self.broadcast({
            "type": "system_status",
            **status
        })


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint handler.

    Protocol:
    - Client connects with client_id in query params
    - Client can send JSON messages to subscribe/unsubscribe from jobs
    - Server sends updates for subscribed jobs

    Client message formats:
    - Subscribe: {"action": "subscribe", "job_id": "xxx"}
    - Unsubscribe: {"action": "unsubscribe", "job_id": "xxx"}
    - Ping: {"action": "ping"}

    Server message formats:
    - Progress: {"type": "progress", "job_id": "xxx", "progress": 0.5, "message": "..."}
    - Status: {"type": "status", "job_id": "xxx", "status": "running"}
    - Complete: {"type": "complete", "job_id": "xxx", "output_video": "...", ...}
    - Error: {"type": "error", "job_id": "xxx", "error": "..."}
    - Queue: {"type": "queue_update", "queue_length": 5, ...}
    - Pong: {"type": "pong"}
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                action = message.get("action", "")

                if action == "subscribe":
                    job_id = message.get("job_id")
                    if job_id:
                        await manager.subscribe_to_job(client_id, job_id)
                        await manager.send_personal(client_id, {
                            "type": "subscribed",
                            "job_id": job_id
                        })

                elif action == "unsubscribe":
                    job_id = message.get("job_id")
                    if job_id:
                        await manager.unsubscribe_from_job(client_id, job_id)
                        await manager.send_personal(client_id, {
                            "type": "unsubscribed",
                            "job_id": job_id
                        })

                elif action == "ping":
                    await manager.send_personal(client_id, {"type": "pong"})

            except json.JSONDecodeError:
                await manager.send_personal(client_id, {
                    "type": "error",
                    "error": "Invalid JSON"
                })

    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await manager.disconnect(client_id)


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager."""
    return manager


# Integration callbacks for JobManager
async def on_job_progress(job_id: str, progress: float, message: str):
    """Callback for job progress updates."""
    await manager.send_progress_update(job_id, progress, message)


async def on_job_status_change(job_id: str, status):
    """Callback for job status changes."""
    await manager.send_status_update(job_id, status.value)

