# Sprint 3 Summary: ML-Based Defect Predictor

## Accomplishments in Sprint 3

1. **Implemented Parallel Processing**:
   - Redesigned the feature extraction pipeline to use multiprocessing for analyzing thousands of commits
   - Significantly improved performance by processing commit batches in parallel
   - Enhanced logging to track batch processing progress

2. **Improved Ground Truth Data**:
   - Expanded ground truth dataset to over 20 files for better model evaluation
   - Added more diverse file types to improve training representation
   - Enhanced model's capability to identify potential defects across a larger codebase

3. **Enhanced Output Management**:
   - Implemented dual output system that displays concise information on console while saving detailed analysis to files
   - Added intelligent truncation for large file lists to improve readability
   - Implemented timestamped output files for better tracking of analysis runs

4. **Output Visualization Improvements**:
   - Streamlined console display to show the most important metrics and results
   - Created a foundation for future visual representations of defect predictions

![Analysis Output Sample](lotsoutput.png)
*Sample of the large number of files predicted as buggy, demonstrating the need for further model refinement*

## Current Challenges

1. **Model Precision**:
   - Current precision is lower than desired, with many false positives
   - Model tends to predict too many files as potentially buggy
   - Even with increased ground truth, the repository size necessitates further ground truth expansion

2. **Output Management in Docker Containers**:
   - Output files are currently saved within Docker containers, making access challenging
   - Need to implement volume mounting for better file access

3. **Large Repository Analysis**:
   - Processing large repositories remains resource-intensive despite parallel processing
   - Current approach limited to analyzing repositories within specific time spans

## Plans for Next Sprint(s)

1. **Multi-Repository Analysis**:
   - Extend the model to analyze multiple repositories simultaneously
   - Remove time-based constraints to get insights across different codebases
   - Develop a more generalized model that works on diverse repository structures

2. **Visualization Enhancements**:
   - Implement interactive visualizations for defect predictions
   - Create dashboards to better interpret model outputs and metrics
   - Develop comparative views to highlight most critical areas for attention

3. **Improved Model Training**:
   - Fine-tune model parameters to reduce false positives
   - Implement feature selection to focus on the most predictive metrics
   - Explore alternative ML approaches beyond Random Forest

4. **Move Beyond Ground Truth**:
   - Begin transitioning to a model that doesn't rely heavily on ground truth data
   - Implement unsupervised learning components to identify patterns without labels
   - Develop confidence metrics that don't depend on predefined defect lists

## Technical Improvements Planned

1. **Docker Configuration Enhancements**:
   - Modify Dockerfile to mount volumes for persistent output storage
   - Improve container resource allocation for faster processing
   - Implement better checkpoint saving for long-running analyses

2. **Feature Extraction Refinement**:
   - Add more sophisticated code complexity metrics beyond simple line counts
   - Implement natural language processing for commit message analysis
   - Add semantic code analysis to better understand code structure

3. **Scalability Improvements**:
   - Implement distributed processing for extremely large repositories
   - Develop incremental analysis capabilities to avoid reprocessing entire repositories
   - Create preprocessing pipelines to reduce memory requirements

## Sample Execution Results

Our improved truncated console output now provides a clear summary of the analysis while saving full details to a file:

```
=== Top 10 Most Likely Buggy Files ===
1. gpt_index/embeddings/base.py - Confidence: 1.0000
2. tests/indices/struct_store/test_base.py - Confidence: 1.0000
3. examples/vector_indices/WeaviateIndexDemo.ipynb - Confidence: 1.0000
4. examples/gatsby/TestGatsby.ipynb - Confidence: 1.0000
5. docs/how_to/data_connectors.md - Confidence: 1.0000
6. examples/langchain_demo/LangchainDemo.ipynb - Confidence: 1.0000
7. examples/paul_graham_essay/TestEssay.ipynb - Confidence: 1.0000
8. examples/cost_analysis/TokenPredictor.ipynb - Confidence: 1.0000
9. tests/mock_utils/mock_text_splitter.py - Confidence: 1.0000
10. gpt_index/composability/__init__.py - Confidence: 1.0000

Total files analyzed: 8440
Files predicted as buggy: 2048
Files predicted as clean: 6392

=== Comparison with Ground Truth ===
Total Files in Ground Truth: 25
Total Files Predicted as Buggy: 2048

=== Model Performance ===
Precision: 0.01
Recall: 0.84
F1 Score: 0.02

False Positives (Files predicted as buggy but not in ground truth):
  - llama-index-integrations/tools/llama-index-tools-shopify/README.md
  - examples/async/AsyncTreeSummarizeQueryDemo.ipynb
  - docs/module_guides/deploying/agents/tools/usage_pattern.md
  - llama_index/program/predefined/evaporate/extractor.py
  - llama-index-integrations/tools/llama-index-tools-azure-translate/llama_index/tools/azure_translate/base.py
  - llama_index/vector_stores/zep.py
  - llama_index/indices/response/accumulate.py
  - tests/agent/react/test_react_output_parser.py
  - gpt_index/readers/milvus.py
  - llama_index/constants.py
  ... 2007 more files (see output.txt for complete list) ...
  - examples/tts/ElevenLabsTTSDemo.ipynb
  - llama-index-integrations/llms/llama-index-llms-alephalpha/llama_index/llms/alephalpha/__init__.py
  - docs/examples/llm/ollama.ipynb
  - docs/examples/customization/llms/SimpleIndexDemo-ChatGPT.ipynb
  - llama_index/evaluation/base.py
  - llama_index/finetuning/__init__.py
  - docs/use_cases/extraction.md
  - docs/understanding/storing/storing.md
  - docs/module_guides/deploying/query_engine/usage_pattern.md
  - llama-index-integrations/readers/llama-index-readers-docugami/llama_index/readers/docugami/base.py

False Negatives (Files in ground truth but not predicted):
  - llama-index-core/llama_index/core/utilities/gemini_utils.py
  - llama-index-core/llama_index/core/node_parser/text.py
  - llama-index-core/llama_index/core/tools/types.py
  - llama-index-core/llama_index/core/llms/openai.py
2025-03-27 20:13:48,629 - INFO - Full output saved to output_20250327_200959.txt
```

The results clearly show that although our model has good recall (0.84), meaning it finds most of the buggy files, its precision is quite low (0.01), indicating many false positives. This demonstrates the need for further model refinement and larger ground truth datasets to improve accuracy.

## Conclusion

Sprint 3 has laid a solid foundation for a more robust and efficient defect prediction system. While we've made significant progress in processing capabilities and output management, we recognize the need for further refinement in model accuracy and repository scalability. Our enhanced ground truth data provides a better baseline, but the sheer scale of modern repositories demands continued improvement in our approach.

By moving toward multi-repository analysis and eventually reducing dependency on ground truth data, we're positioning the system to provide more valuable insights with less manual preparation. The future development will focus on making the tool more practical for real-world development teams while applying the software engineering principles covered in class. 
