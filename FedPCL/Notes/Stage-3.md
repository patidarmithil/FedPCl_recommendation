
---
In **Stage 3**, the system focuses on **Personalization**. We aren't doing the "vibe check" (Contrastive Learning) or "secret ink" (LDP) yet. Think of this as the foundation where we teach the system how to give students (users) a syllabus tailored to their specific "Study Group."

Here is how the Stage 3 files work together:

---

### 1. `train_stage3.py` — The Project Director

This is the only file you actually "run" to start the system. It acts as the director of the entire operation.

- **What it does:** It handles the command-line arguments (like which dataset to use) and sets the "Rules of the School" (Hyperparameters).
    
- **Key Settings:** It defines things like how many rounds to study (`n_rounds`), how many clusters to form (`n_clusters`), and the learning rates.
    
- **Connection:** It calls the `train_stage3()` function inside the Federated Core to start the engine. 

### 2. `federated_core_stage3.py` — The Classroom Conductor

If `train_stage3.py` is the director, this is the teacher in the room making sure every "Communication Round" happens on time.

- **What it does:** * It uses `data_loader.py` to bring in the books and student lists.
    
    - It creates one `ServerStage3` object and thousands of `ClientStage3` objects.
        
    - It runs the main loop: it picks 128 students each round, gives them homework, and collects their results.
        
- **The Big Fix:** It ensures that **every** student is sorted into a study group (cluster) every 10 rounds, not just the ones currently studying.
    
- **Connection:** This is the "middle-man" that coordinates between the Server and all the Clients.
    

### 3. `server_stage3.py` — The Master Librarian

The server's job is to maintain the "Master Syllabus" and organize the "Study Groups".

- **What it does:**
    
    - **Master Lists:** It holds `E_global` (the main list for everyone) and `E_clusters` (specialized lists for the 5 study groups).
        
    - **K-Means Clustering:** It looks at the "vibe" (embeddings) sent by students and uses math to group similar students together.
        
    - **Aggregation:** When students send back their "deltas" (their suggested changes), the librarian updates the Master Syllabus so it gets better for the next round.
        
- **Math Power:** It calculates the **Personalized Syllabus** ($E_{personal}$) for each student using a mix: $E_{personal} = \mu_1 E_{cluster} + \mu_2 E_{global}$.
    

### 4. `client_stage3.py` — The Individual Student

This is the most active part of the code. Each client represents one user studying privately at home.

- **What it does:**
    
    - **Local Training:** It takes the personalized syllabus from the teacher and runs a "mini-GNN" (LightGCN) on its own private books.
        
    - **Learning:** It uses **BPR Loss** to learn which books the user likes more than others.
        
    - **Reporting:** After studying, it doesn't send its private book list. It only sends **"deltas"**—basically saying, "Hey teacher, based on my syllabus, I think these items should be moved up or down in importance".
        
- **Connection:** It receives $E_{personal}$ from the server and sends back `item_deltas` and its current `user_emb` (vibe).
    

---

### How they are connected in the workflow:

1. **Start:** You run `train_stage3.py`.
    
2. **Setup:** `federated_core_stage3.py` loads the data and creates the Librarian (`server`) and the Students (`clients`).
    
3. **Clustering:** The `server` groups all students into 5 clubs.
    
4. **Round Starts:** The `core` picks 128 students.
    
5. **Homework:** Each selected student (`client`) gets a personalized syllabus from the `server`.
    
6. **Study:** The `client` runs 5 epochs of training on its own device.
    
7. **Return:** The `client` sends the smudged "deltas" back to the `server`.
    
8. **Update:** The `server` averages all those deltas and updates the Master Syllabus.
    
9. **Repeat:** This happens 400 times until the recommendations are very accurate.