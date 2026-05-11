**Stage 5** is the "Full FedPCL" system. It takes everything from Stage 4 (Personalization + Neighbors + Vibe Checks) and adds the **"Secret Ink" (Local Differential Privacy)** to ensure that no one—not even the teacher—can peek at a student's private diary by analyzing their updates.

Here is the breakdown of the final stage files:

---

### 1. `train_stage5.py` — The Director with a Privacy Policy

This is the final "master" file you run to execute the complete system.

- **What it does:** It introduces the privacy settings that define how thick the "secret ink" should be.
    
- **Key Parameters:**
    
    - **`clip_sigma` ($\sigma$):** The boundary. If a student's update is too loud or extreme, it gets "clipped" down to this size to prevent it from standing out.
        
    - **`lambda_laplace` ($\lambda$):** The amount of random noise (ink) added to the report.
        
- **Privacy Budget ($\epsilon$):** It calculates $\epsilon = \sigma / \lambda$. A smaller $\epsilon$ means more privacy but a bigger drop in recommendation accuracy.
    

### 2. `client_stage5.py` — The Student with the Ink Bottle

This file inherits all the neighborhood learning from Stage 4 but adds a final "masking" step before sending homework back.

- **What it does:** It contains the `apply_ldp` function.
    
- **The Process:** After the student finished studying their books and comparing vibes with neighbors, they take their `item_deltas` and `user_emb` and smudge them with Laplacian noise.
    
- **Connection:** This ensures that the data leaving the student's device is mathematically "fuzzy".
    

### 3. `server_stage5.py` — The Unchanged Librarian

This file is actually identical to `server_stage4.py`.

- **What it does:** It re-exports the Stage 4 server logic.
    
- **Why?** Because the "Secret Ink" is applied by the students _before_ the teacher sees the reports. The Librarian just aggregates the noisy reports the same way they did before.
    

### 4. `federated_core_stage5.py` — The Master Conductor

This is the most advanced version of the training loop. It includes extra reporting to see how the privacy noise is affecting the "Final Grade".

- **What it does:** * **Paper Comparison:** It automatically compares your current HR@10 and NDCG@10 scores against the original **Paper Targets** (like the 80.36% for the Steam dataset).
    
    - **Score Deviation:** At the end of training, it prints a "Verdict" to tell you if your results match the paper's claims within a 2% or 5% margin.
        
    - **Snapshots:** It automatically saves the "Master Syllabus" (embeddings) at Round 1 and Round 400 so you can visualize how the library evolved.
        
- **Connection:** It pulls together the data loader, the noisy clients, and the server to run the full federated experiment.
    

---

### How the Full System Works Together:

1. **Initialization:** `train_stage5.py` starts the engine with your chosen privacy level ($\epsilon$).
    
2. **Neighborhoods:** The `server` maps out 2-hop neighbors so students aren't studying in isolation.
    
3. **Local Learning:** Students (`client`) use **BPR Loss** for books and **Contrastive Loss** (from `contrastive.py`) for vibes.
    
4. **Privacy Shield:** Before uploading, the `client` smudges their results using the LDP settings from `train_stage5.py`.
    
5. **Aggregation:** The `server` averages these noisy updates to improve the global models.
    
6. **Verdict:** After 400 rounds, `federated_core_stage5.py` tells you exactly how much accuracy you sacrificed for privacy compared to the original research.