"""
BOM Part Number NER Model
Transformer-based Named Entity Recognition model for extracting Part Numbers from BOM data
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional, Dict


class BOMPartNumberNER(PreTrainedModel):
    """Part Number 추출용 NER 모델"""

    def __init__(self, config, model_name: str = 'bert-base-uncased', num_labels: int = 3):
        """
        Args:
            config: Model configuration
            model_name: Pretrained transformer model name
            num_labels: Number of labels (default: 3 for O, B-PART, I-PART)
        """
        super().__init__(config)
        self.num_labels = num_labels

        # Label mappings
        self.label2id = {'O': 0, 'B-PART': 1, 'I-PART': 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Transformer backbone
        self.transformer = AutoModel.from_pretrained(model_name, config=config)

        # Dropout for regularization
        classifier_dropout = getattr(config, 'classifier_dropout', None)
        if classifier_dropout is None:
            classifier_dropout = getattr(config, 'hidden_dropout_prob', 0.1)
        if classifier_dropout is None:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)

        # Classification head
        hidden_size = config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            labels: Ground truth labels (batch_size, seq_length)
            return_dict: Whether to return dictionary
        
        Returns:
            Dictionary with loss and logits
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get sequence output (last hidden state)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Classification
        logits = self.classifier(sequence_output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )


def create_model(model_name: str = 'bert-base-uncased', num_labels: int = 3) -> BOMPartNumberNER:
    """
    Factory function to create BOM NER model
    
    Args:
        model_name: Name of pretrained transformer model
        num_labels: Number of labels
    
    Returns:
        Initialized BOMPartNumberNER model
    """
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    
    model = BOMPartNumberNER(config, model_name=model_name, num_labels=num_labels)
    return model
