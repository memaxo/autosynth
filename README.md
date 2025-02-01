# AutoSynth

AutoSynth is a modular synthetic dataset generation pipeline that automates the process of collecting, processing, and generating high-quality synthetic data. It uses advanced LLM techniques and semantic analysis to ensure data quality and relevance.

## Features

- 🔄 **Multi-Stage Pipeline**
  - Discovery: Smart URL collection using search APIs
  - Collection: Multi-format document loading and caching
  - Processing: Quality assessment and content enhancement
  - Generation: Synthetic data creation using DeepSeek R1

- 📊 **Project Management**
  - Multiple project support
  - State persistence
  - Progress tracking
  - Metric monitoring

- 🛠 **Advanced Processing**
  - LLM-based quality assessment
  - Content enhancement
  - Semantic analysis
  - Vector store integration
  - Batch processing

- 🔍 **Quality Control**
  - Relevance checking
  - Quality scoring
  - Duplicate detection
  - Content validation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AutoSynth.git
cd AutoSynth

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-api-key"
export KLUSTER_API_KEY="your-kluster-api-key"
```

## Usage

```bash
# Create a new project
autosynth --new-project "defensive cybersecurity"

# List all projects
autosynth --list-projects

# Run specific stage with target metrics
autosynth --project-id abc123 --stage discover --target-urls 200
```

## Project Structure

```
~/.autosynth/
├── projects.json           # Project metadata
├── autosynth.log          # Main log file
└── [project_id]/          # Project directories
    ├── state.json         # Project state
    ├── discover/          # Stage data
    ├── collect/          
    ├── process/
    └── generate/
```

## Dependencies

- LangChain: LLM integration and chains
- Curator: Content curation
- Kluster.ai: DeepSeek R1 integration
- DuckDuckGo API: URL discovery
- aiohttp: Async HTTP requests
- beautifulsoup4: HTML parsing
- PyPDF2: PDF processing
- gitpython: Git repository handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 