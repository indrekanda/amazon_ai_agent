## AI Agenf for Amazon products
---
### Usage

To run locally:
1. Clone: `git clone https://github.com/indrekanda/amazon_ai_agent.git`
2. Create venv: `uv sync` or `uv sync --extra dev`
3. Activate: `source .venv/bin/activate`
4. Copy .env to the root dir
5. Run the app: `make run-docker-compose` (make sure Docker is up)


---
### UI
Streamlit: http://localhost:8501/  
LangSmith: https://smith.langchain.com/  
Qdrant: http://localhost:6333/dashboard  
FastAPI: http://localhost:8000/docs  

---
### Acknowledgements and Citations

This repository utilizes data from the following research paper. If you use this repository and its data, please consider citing the original source of the data:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
The data used in this repository is the Amazon Reviews 2023 dataset. The official website: https://amazon-reviews-2023.github.io/main.html

