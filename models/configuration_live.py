from transformers import PretrainedConfig


class LiveConfigMixin(PretrainedConfig):
    def __init__(
        self,
        *,
        stage: int = 2,
        model_variant: str = None,
        vision_pretrained: str = None,
        connector_type: str = None,
        compressed_tokens: int = 0,
        hand_tokens: int = 0,
        object_tokens: int = 0,
        box_loss_weight: float = 1.0,
        connector_nhead: int = None,
        connector_num_layers: int = None,
        connector_hidden_dim: int = None,
        pretrained_ckpt_path: str = None,
        frame_resolution: int = None,
        frame_token_cls: bool = None,
        frame_token_pooled: int = None,
        frame_num_tokens: int = None,
        v_placeholder: str = "<v>",
        long_placeholder: str = "<long",
        frame_token_interval: str = None,
        v_placeholder_id: int = None,
        frame_token_interval_id: int = None,
        long_placeholder_id: int = None,
        N_s: int = 32,
        N_l: int = 0,
        vision_hidden_size: int = None,
        vision_num_tokens: int = None,
        stream_loss_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stage = stage
        self.model_variant = model_variant
        self.vision_pretrained = vision_pretrained
        self.connector_type = connector_type
        self.compressed_tokens = compressed_tokens
        self.hand_tokens = hand_tokens
        self.object_tokens = object_tokens
        self.box_loss_weight = box_loss_weight
        self.connector_nhead = connector_nhead
        self.connector_num_layers = connector_num_layers
        self.connector_hidden_dim = connector_hidden_dim
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.frame_resolution = frame_resolution
        self.frame_token_cls = frame_token_cls
        self.frame_token_pooled = frame_token_pooled
        self.frame_num_tokens = frame_num_tokens
        self.v_placeholder = v_placeholder
        self.long_placeholder = long_placeholder
        self.frame_token_interval = frame_token_interval
        self.v_placeholder_id = v_placeholder_id
        self.long_placeholder_id = long_placeholder_id
        self.frame_token_interval_id = frame_token_interval_id
        self.N_s = N_s
        self.N_l = N_l
        self.vision_hidden_size = vision_hidden_size
        self.vision_num_tokens = vision_num_tokens
        self.stream_loss_weight = stream_loss_weight
