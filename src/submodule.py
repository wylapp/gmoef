import torch
from torch import nn
from transformers import AutoConfig, AutoModel

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.bert_dim, config.bert_dim)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.bert_dim, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Bert_Layer(torch.nn.Module):
    def __init__(self, args):
        super(Bert_Layer, self).__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.bert_path)
        self.config.output_hidden_states = True
        self.config.num_hidden_layers = args.num_hidden_layers
        # self.config.hidden_dropout = 0.
        # self.config.hidden_dropout_prob = 0.
        # self.config.attention_dropout = 0.
        # self.config.attention_probs_dropout_prob = 0.
        self.bert_layer = AutoModel.from_pretrained(args.bert_path, config=self.config)

    def forward(self, ids, masks):
        bert_output = self.bert_layer(input_ids=ids, attention_mask=masks, output_hidden_states=True)

        return bert_output
    
    def freeze_partial(self):
        for i in range(self.args.freeze_lower):
            for p in self.bert_layer.encoder.layer[i].parameters():
                p.requires_grad = False
    
    def unfreeze_partial(self):
        for name ,param in self.bert_layer.named_parameters():
            param.requires_grad = True

class PWLayer(nn.Module):
    """
    Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)
    
class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.

    Update on Jan. 5th: Change to pytorch provided multihead attention for better performance
    and effecient memory consumption. By Youlin
    """

    def __init__(self, pooler_type, config):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
            "attention",
            "gate"
        ], (
            "unrecognized pooling type %s" % self.pooler_type
        )
        self.config = config
        if self.pooler_type == 'attention':
            self.norm = nn.LayerNorm(self.config.word_embedding_dim)
            self.mh_att = nn.MultiheadAttention(self.config.word_embedding_dim, 6)
            self.att_layer = AdditiveAttention(self.config.word_embedding_dim, self.config.news_dim)

        if self.pooler_type == 'gate':
            self.norm = nn.LayerNorm(self.config.word_embedding_dim)
            self.mh_att = nn.MultiheadAttention(self.config.word_embedding_dim, 6, 128, 128, self.config.enable_gpu)
            self.fc1 = nn.Linear(768, 300)
            self.fc2 = nn.Linear(300, 1, False)

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            if self.pooler_type == 'cls_before_pooler':
                return last_hidden[:, 0]
            else:
                return pooler_output
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "attention":
            token_vectors = hidden_states[-1]
            att_result, _ = self.mh_att(token_vectors, mask=attention_mask)
            att_result = self.norm(token_vectors + att_result)
            pooled_result = self.att_layer(token_vectors, attention_mask)

            return pooled_result
        elif self.pooler_type == "gate":
            token_vectors = hidden_states[-1]
            pooled_result, _ = self.mh_att(token_vectors, mask=attention_mask)

            h_tilde = self.norm(token_vectors + pooled_result)
            r = torch.softmax(self.fc2(torch.tanh(self.fc1(h_tilde))), dim=1)
            h = torch.bmm(r.transpose(-1, -2), h_tilde).squeeze(dim=1)
            return h
        
class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg: 
        d_h: the last dimension of input
    '''
    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x