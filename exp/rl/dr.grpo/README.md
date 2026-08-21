# Difference between Dr.GRPO and vanilla GRPO
Dr.GRPO, i.e., GRPO Done Right, is a variant of GRPO that incorporates several modifications to enhance its performance. The key differences between Dr.GRPO and vanilla GRPO are as follows:

1. loss aggregation mode: from `seq-mean-token-mean` to `seq-mean-token-sum`. 删除长度归一 1/|O|，因为这个会对长序列和短序列的token惩罚奖励力度不一致(短序列的token更少，被更新的程度更大，长序列的token更多，被更新的程度更小)，导致正确序列偏向短（正确时，从token角度看，短序列被更新的强度>长序列），错误序列偏向长 (错误时，短序列被惩罚的强度>长序列)。所以 token mean 去除归一化项 1/|O| 后变成 token sum。
2. norm adv by std in grpo: 删除std，因为在那些接近全0和全1的组里，std很低，导致adv被夸张的放大，进而导致更新过大，训练不稳定。删除std后，adv=reward - baseline，reward和baseline都是0-1之间的数，所以adv的范围也在-1到1之间，不会被夸张放大，训练更稳定。

对应 verl 修改为:
1. actor_rollout_ref.actor.loss_agg_mode: 从 "seq-mean-token-mean" 改为 "seq-mean-token-sum"
2. algorithm.norm_adv_by_std_in_grpo: 从 True 改为 False
