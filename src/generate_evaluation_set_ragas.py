import json
import logging
import sys
import time
import asyncio
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to ensure imports work
sys.path.append(str(Path(__file__).parent.parent))

# Initialize logging
LOG_FILE = Path('assets/evaluation_ragas.log')
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    ]
)
logger = logging.getLogger(__name__)

try:
    from RAG_pipeline.responder import Responder
    from src.query_handler import QueryHandler
except ImportError as e:
    logger.error(f"Import failed: {e}")
    raise

class EvaluationGenerator:
    def __init__(self):
        logger.info("Initializing EvaluationGenerator")
        try:
            self.responder = Responder()
            self.query_handler = QueryHandler()
            logger.info("Components initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize components")
            raise

    def load_queries(self, filepath: Path) -> list:
        """Load just the queries from golden_set_ragas.json"""
        try:
            logger.info(f"Loading queries from {filepath}")

            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            with filepath.open('r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"Expected list, got {type(data)}")

            # Extract just the query fields
            queries = [item.get('query', '').strip() for item in data]
            queries = [q for q in queries if q]  # Filter out empty queries

            if not queries:
                logger.warning("No valid queries found in golden_set.json")

            logger.info(f"Loaded {len(queries)} queries")
            return queries

        except Exception as e:
            logger.exception("Failed to load queries")
            raise

    async def process_queries(self, queries: list) -> list:
        """Process queries through RAG pipeline asynchronously"""
        if not queries:
            logger.warning("Empty query list provided")
            return []

        model_outputs = []
        logger.info(f"Starting processing of {len(queries)} queries")

        for i, query in enumerate(tqdm(queries, desc="Processing")):
            try:
                if not query.strip():
                    logger.warning(f"Skipping empty query at index {i}")
                    continue

                # FAISS similarity search to get relevant context
                results = await asyncio.to_thread(self.query_handler.process_query, query)
                context = "\n\n".join([res[2] for res in results])

                # Generate response using the context
                response = await asyncio.to_thread(self.responder.generate_response, query, context)

                model_outputs.append({
                    "query": query,
                    "context": context,
                    "output": response
                })

                await asyncio.sleep(15)  # Replaces time.sleep for async rate limiting

            except Exception as e:
                error_msg = f"Error on query {i+1}: {str(e)}"
                logger.error(error_msg)
                model_outputs.append({
                    "query": query,
                    "context": "",  # Store empty context on error
                    "output": f"Error: {str(e)}"
                })

        logger.info(f"Completed processing. Generated {len(model_outputs)} responses")
        return model_outputs

    def _write_json(self, path, data):
        """Helper function to write JSON synchronously."""
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    async def save_outputs(self, outputs: list, filepath: Path):
        """Save outputs with verification asynchronously"""
        try:
            if not outputs:
                logger.error("No outputs to save!")
                return False

            filepath.parent.mkdir(exist_ok=True)
            temp_path = filepath.with_suffix('.tmp')
            logger.info(f"Writing to temporary file: {temp_path}")

            await asyncio.to_thread(self._write_json, temp_path, outputs)
            await asyncio.to_thread(temp_path.replace, filepath)

            logger.info(f"Successfully saved {len(outputs)} records to {filepath}")
            return True

        except Exception as e:
            logger.exception("Failed to save outputs")
            return False

async def main():
    try:
        # Path setup
        assets_dir = Path('assets')
        golden_set_path = assets_dir / 'golden_set_ragas.json'
        output_path = assets_dir / 'model_output_ragas.json'

        # Initialize
        generator = EvaluationGenerator()

        # Load just the queries (ignoring ground truth answers)
        queries = generator.load_queries(golden_set_path)

        # Process with terminal output
        model_outputs = await generator.process_queries(queries)

        # Save
        if not await generator.save_outputs(model_outputs, output_path):
            raise RuntimeError("Failed to save outputs")

        print(f"\nSUCCESS: Results saved to {output_path}")
        print(f"Detailed log: {LOG_FILE}")
        return 0

    except Exception as e:
        logger.exception("Fatal error in main")
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
