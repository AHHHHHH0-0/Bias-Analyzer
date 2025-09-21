import json
import os
from datetime import datetime


class OutputFormatter:
    """
    Formats and saves prediction results.
    """
    
    def __init__(self, output_config):
        """
        Initialize the output formatter.
        
        Args:
            output_config (dict): Output configuration
        """
        self.output_file = output_config["output_file"]
        self.show_in_terminal = output_config["show_in_terminal"]
        self.terminal_display_limit = output_config["terminal_display_limit"]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
    
    def format_results(self, results, summary, model_info, time):
        """
        Format results into a structured output.
        
        Args:
            results (list): Prediction results
            summary (dict): Prediction summary
            model_info (dict): Model information
            time (time): Total inference time
            
        Returns:
            dict: Formatted output
        """
        formatted_output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_info": model_info,
                "total_articles": len(results),
                "total_time": f"{time:.2f} seconds",
                "summary": summary
            },
            "predictions": results
        }
        
        return formatted_output
    
    def save_to_json(self, formatted_output):
        """
        Save results to JSON file.
        
        Args:
            formatted_output (dict): Formatted results
        """
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_output, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"✗ Failed to save results: {str(e)}")
    
    def display_in_terminal(self, formatted_output):
        """
        Display results in terminal.
        
        Args:
            formatted_output (dict): Formatted results
        """
        if not self.show_in_terminal:
            return
        
        metadata = formatted_output["metadata"]
        results = formatted_output["predictions"]
        summary = metadata["summary"]
               
        # Display metadata
        model_info = metadata["model_info"]
        print()
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Model: {model_info['model_name']} ({model_info['layers']}, {model_info['training']})")
        print(f"Total Articles: {metadata['total_articles']}")
        print(f"Total Time: {metadata["total_time"]}")
        print(f"Results saved to: {self.output_file}")

        # Show first n individual result
        self._display_individual(results)
        
        # Show summary
        self._display_summary(summary)

    def _display_individual(self, results):
        """
        Display the first n results.

        Args:
            results (list): Prediction results
        """
        # Handle less articles than display limit
        length = len(results) if self.terminal_display_limit > len(results) else self.terminal_display_limit

        print(f"\nINDIVIDUAL PREDICTIONS")
        print("-" * 40)
        print(f"(Showing first {length} of {len(results)} articles)") 
        
        for i, result in enumerate(results[:length]):
            article = result["article_data"]
            prediction = result["prediction"]
            
            # Show title, source, prediction (with confidence)
            title = article.get("title", "No title")
            source = article.get("source", "Unknown source")
            print(f"\n[{i+1}] {title}")
            print(f"    Source: {source}")
            print(f"    Prediction: {prediction['label'].upper()} (confidence: {prediction['confidence']:.3f})")
            
            # Show probability breakdown
            probs = prediction['probabilities']
            prob_str = " | ".join([f"{label}: {prob:.3f}" for label, prob in probs.items()])
            print(f"    Probabilities: {prob_str}")
        
        # Remaining
        if len(results) > length:
            remaining = len(results) - length
            print(f"\n... and {remaining} more articles")

    def _display_summary(self, summary):
        """
        Display summary of inference results.

        Args:
            summary (dict): Prediction summary
        """
        # Display summary statistics
        print(f"\nSUMMARY STATISTICS")
        print("-" * 40)
        # Label distribution
        print("Label Distribution:")
        for label, count in summary["label_distribution"].items():
            percentage = (count / summary["total_articles"]) * 100
            print(f"  {label.upper()}: {count} ({percentage:.1f}%)")
        # Confidence statistics
        conf_stats = summary["confidence_stats"]
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {conf_stats['mean']:.3f}")
        print(f"  Std:  {conf_stats['std']:.3f}")
        print(f"  Min:  {conf_stats['min']:.3f}")
        print(f"  Max:  {conf_stats['max']:.3f}")
        
        print(f"\nConfidence Levels:")
        print(f"  High (>0.8): {summary['high_confidence_count']} articles")
        print(f"  Low (<0.5):  {summary['low_confidence_count']} articles")
    
    def process_output(self, results, summary, model_info, time):
        """
        Process and save all outputs.
        
        Args:
            results (list): Prediction results
            summary (dict): Prediction summary
            model_info (dict): Model information
        """
        # Format results
        formatted_output = self.format_results(results, summary, model_info, time)
        
        # Save to JSON
        self.save_to_json(formatted_output)
        
        # Display in terminal
        self.display_in_terminal(formatted_output)
