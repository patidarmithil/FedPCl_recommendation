In **Stage 4**, we move from just having general study groups (Stage 3) to actually talking to your "neighbors"—people who liked similar books—to solve the problem of having a very small personal notebook (Data Sparsity).

Stage 4 adds **Structural Contrastive Learning** (the "vibe check") and **Graph Expansion** (the "neighborhood"). Here is how the files handle this:

---

### 1. `train_stage4.py` — The Director (with New Goals)

This is still your main entry point.

- **What it does:** It adds new "Rules of the School" specifically for the vibe check:
    
    - **$\beta_1$ (beta1):** How much you care about the vibe check versus just learning the books.
        
    - **$\tau$ (tau):** How strictly you define "similar vibes".
        
    - **Warmup Rounds:** It tells the students to just focus on books for the first 20 rounds before starting the vibe check.
        

### 2. `server_stage4.py` — The Librarian (with a Social Map)

The Librarian now tracks who is reading what in a more detailed way to help students find their neighbors.

- **What it does:** * **`item2users` index:** It maintains a map of which students have read which books.
    
    - **Neighborhood Finder:** When a student asks, the Librarian looks at the map and says, "Student B and C have read the same books as you; they are your 2-hop neighbors".
        
    - **Neighbor Embs:** It gathers the "anonymous vibes" (embeddings) of those neighbors and sends them to you so you can compare.
        

### 3. `client_stage4.py` — The Student (with a Bigger Subgraph)

The student is now much busier. Instead of just looking at their own notebook, they look at their neighbors' interests too.

- **What it does:** * **Precomputed Weights:** To stay fast, it calculates the "influence" of its neighbors only once at the beginning.
    
    - **Expanded LightGCN:** It runs the GNN not just on its 10 books, but on the 20-30 books its neighbors like as well.
        
    - **The Vibe Comparison:** It checks if its "even-layer" learning (higher-order structural patterns) matches its "layer-0" (original) data.
        

### 4. `contrastive.py` — The Vibe-Check Specialist

This is a new file that acts like a specialized calculator for the "vibe check" math.

- **What it does:** * **User CL (Eq. 5):** It calculates how close you are to your neighbors' tastes.
    
    - **Item CL (Eq. 6):** It makes sure the "vibe" of a specific book stays consistent even if different neighbors are reading it.
        
    - **Variance Guard:** It has a safety feature: if you and your neighbors are all reading the exact same thing (no variety), it skips the vibe check to avoid getting confused.
        

### 5. `federated_core_stage4.py` — The Conductor (with Social Connections)

- **What it does:** In every round, it asks the Librarian to find the neighbors for the 128 selected students.
    
- **Connection:** It passes the `neigh_embs` (the neighbors' vibes) from the `server` to the `client`.
    

---

### How Stage 4 connects while running:

1. **Round Starts:** The `core` picks 128 students.
    
2. **Find Neighbors:** The `server` looks at its map and finds 2-hop neighbors for those students.
    
3. **Download Vibes:** The students download their personalized syllabus AND the vibes of their neighbors.
    
4. **Study & Compare:** The student uses `contrastive.py` to make sure their learning isn't just about their 5 books, but fits the "vibe" of their neighborhood.
    
5. **Return:** The student sends back the results to update the global syllabus