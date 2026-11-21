#!/usr/bin/env bash
# Ablations (each trains its own CaloNSF; JEPA encoder is shared from the main run where relevant).
# Usage: scripts/run_ablations.sh configs/experiments/calochallenge.yaml
set -euo pipefail
CFG=$1
PY=${PY:-.venv/bin/python}
BASE=$($PY -c "from calonsf.utils.config import load_config; c=load_config('$CFG'); print(c.out_dir + '/' + c.exp_name)")
ENC=$BASE/jepa/encoder.pt
run () {  # name, overrides...
  local name=$1; shift
  echo "=== ablation: $name"
  $PY scripts/train_sr.py --config "$CFG" -o exp_name=abl_$name "$@"
  $PY scripts/evaluate.py --config "$CFG" -o exp_name=abl_$name eval.baselines=[] "$@"
}
run no_jepa            train.pretrained_encoder=null
$PY scripts/pretrain_jepa.py --config "$CFG" -o exp_name=abl_no_cross_scale jepa.cross_scale=false
run no_cross_scale     train.pretrained_encoder=auto
run one_shot           model.ladder=one_shot train.pretrained_encoder=$ENC
run no_conservation    model.conservation=false train.pretrained_encoder=$ENC
run no_occupancy       model.use_occupancy=false train.pretrained_encoder=$ENC
run regression         model.generator=regression train.pretrained_encoder=$ENC
run absolute_pos       model.pos_encoding=absolute train.pretrained_encoder=null
run no_profile_loss    loss.profile=0.0 train.pretrained_encoder=$ENC
run no_jepa_loss       loss.jepa=0.0 train.pretrained_encoder=$ENC
run conditioning_off   model.conditioning.incident_energy=false train.pretrained_encoder=null
# resolution generalisation: train on <=x4 only, evaluate at x8
run train_le4_test_x8  data.lr_factors=[2,4] eval.lr_factors=[8] train.pretrained_encoder=$ENC
