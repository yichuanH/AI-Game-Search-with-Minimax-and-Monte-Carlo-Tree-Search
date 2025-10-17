Defeat Lin: AI Game Search with Minimax/MCTS for a Hex-Grid “Strands”-Style Game

> NYCU Artificial Intelligence (Spring 2024) – Project 2

<img width="355" height="342" alt="截圖 2025-10-17 下午2 19 32" src="https://github.com/user-attachments/assets/30244ae1-6ba2-4067-a3be-ca0071d69875" />

This project implements a hex-grid, turn-based game (inspired by the “Strands” assignment story) and an AI agent powered by **Monte Carlo Tree Search (MCTS)**.  
You can play **Human vs Human**, **Human vs AI (MCTS)**, or **AI vs AI** and measure strategy effectiveness via the final connected-component score.

- UI / game loop built with **Pygame**. :contentReference[oaicite:6]{index=6}  
- Core search implemented via **MCTS** with UCB and rollout simulation. :contentReference[oaicite:7]{index=7}

---

## Demo

- Hex grid with labeled cells (1,2,3,5,6) drawn and clickable.
- A turn indicator and an **End Turn** button in the UI. :contentReference[oaicite:8]{index=8}

---

## Rules (Simplified)

- The board is a hexagon-shaped grid. Each cell has a **label** (1/2/3/5/6) roughly based on its ring/position. :contentReference[oaicite:9]{index=9}
- **Round 1**: the current player must select **one** cell labeled **2**.  
- **Round ≥ 2**: the first clicked cell in that round sets the **label n** for the round, and the player must select **n** cells of that same label in the round. :contentReference[oaicite:10]{index=10}
- Players alternate **black → white → black → …**  
- Game ends when all cells are selected. The winner is the player with the **largest connected component** of their color. (6-neighbor connectivity on the hex grid.) :contentReference[oaicite:11]{index=11}

---


How to Run
### Human vs AI(MCTS)
python main_r6.py human random

### AI(MCTS) vs AI(MCTS)
python main_r6.py random random

### Human vs Human
python main_r6.py human human
