"""Inference server to handle requests for model inference in parallel."""

import multiprocessing as mp

import numpy as np
import torch

class InferenceServer:
    """Inference server to handle requests for model inference in parallel."""

    def __init__(self, model, request_queue, response_queues):
        self.model = model
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.server_process = mp.Process(
            target=self._handle_inference_requests,
            args=(self.request_queue, self.response_queues, self.model)
        )

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

            state_gpu = torch.tensor(
                np.array(states),
                dtype=torch.float32
            ).to(model.device)

            self.model.eval()
            with torch.no_grad():
                policy_logits, value = self.model(state_gpu)
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
