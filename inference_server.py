"""Inference server to handle requests for model inference in parallel."""

import multiprocessing as mp
import asyncio

import numpy as np
import torch

class InferenceServer:
    """Synchronous inference server to handle requests for model inference in parallel processes."""

    def __init__(self, model, request_queue, response_queues):
        self.model = model
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.server_process = mp.Process(
            target=self._handle_inference_requests,
            args=(self.request_queue, self.response_queues, self.model)
        )

    def set_model(self, model):
        """Set the model for the inference server to use for evaluation."""
        self.model = model

    def start(self):
        """Start the inference server in a separate process."""
        self.server_process.start()

    def stop(self):
        """Stop the inference server."""
        # None is the shutdown signal
        self.request_queue.put(None)
        self.server_process.join()
        self.request_queue.close()
        for q in self.response_queues.values():
            q.close()
        self.request_queue.join_thread()
        for q in self.response_queues.values():
            q.join_thread()

    def _handle_inference_requests(self, request_queue, response_queues, model):
        """
        Manage model inference requests from multiple workers,
        process them using the model and send back the response
        """

        while True:

            requests = []
            states = []

            # Collect requests from the request queue until a timeout occurs
            while True:
                try:
                    request = request_queue.get(timeout=1e-4)
                    # Shutdown signal None received
                    if request is None:
                        return
                    requests.append(request)
                    states.append(request['state'])
                except mp.queues.Empty:
                    if requests:
                        break

            states_gpu = torch.tensor(
                np.array(states),
                dtype=torch.float32
            ).to(model.device)

            self.model.eval()
            with torch.no_grad():
                policy_logits, value = self.model(states_gpu)
                policy_logits = policy_logits.cpu().numpy()
                value = value.cpu().numpy()

            # Send responses to the queues corresponding to the
            # worker ids of the sent requests
            for req, pol, val in zip(requests, policy_logits, value):
                worker_id = req['worker_id']
                response_queues[worker_id].put({
                    'policy_logits': pol,
                    'value': val.item()
                })

class AsyncInferenceServer():
    """Asynchronous inference server to handle requests for model inference asynchronously."""

    def __init__(
        self,
        model,
        request_queue: asyncio.Queue,
        response_queues: dict[int, asyncio.Queue]
    ):
        self.model = model
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.server_task = None

    def set_model(self, model):
        """Set the model for the inference server to use for evaluation."""
        self.model = model

    def start(self):
        """Start the asynchronous inference server."""
        self.server_task = asyncio.create_task(
            self._handle_inference_requests_async(
                self.request_queue, self.response_queues, self.model
            )
        )

    async def stop(self):
        """Stop the asynchronous inference server."""
        # None is the shutdown signal
        await self.request_queue.put(None)
        await self.server_task

    async def _handle_inference_requests_async(self, request_queue, response_queues, model):
        """
        Manage model inference requests from asynchronous workers in a single process
        """

        loop = asyncio.get_running_loop()

        while True:

            requests = []
            states = []

            # Collect requests from the request queue until a timeout occurs
            while True:
                try:
                    request = await asyncio.wait_for(request_queue.get(), timeout=1e-4)
                    # Shutdown signal None received
                    if request is None:
                        return
                    requests.append(request)
                    states.append(request['state'])
                except asyncio.TimeoutError:
                    if requests:
                        break

            policy_logits, values = await loop.run_in_executor(
                None,
                self._model_inference,
                model,
                states
            )

            # Send responses to the queues corresponding to the
            # worker ids of the sent requests
            for req, pol, val in zip(requests, policy_logits, values):
                worker_id = req['worker_id']
                await response_queues[worker_id].put({
                    'policy_logits': pol,
                    'value': val.item()
                })

    def _model_inference(self, model, states):
        """
        Perform model inference on the given states.
        This is a helper method to encapsulate the inference logic.
        """
        states_gpu = torch.tensor(
            np.array(states),
            dtype=torch.float32
        ).to(model.device)
        model.eval()
        with torch.no_grad():
            policy_logits, value = model(states_gpu)
            return policy_logits.cpu().numpy(), value.cpu().numpy()
