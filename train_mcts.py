"""Script for running the training of the MCTS agent."""

from mcts_agent import MCTSAgent

if __name__ == "__main__":
    agent = MCTSAgent()
    agent.train()
