# Risk-event-tag-recognition-based-on-large-scale-pre-training-Model-
| 模型描述                                    | dev macro-F1        | 线上得分        |备注 |
| -------------------------------------------      | ------------------- |------------------- | ------------------- |
| nezha_base_wwm_dice_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan0.2_v3 | 0.5511  | 0.5489  |     |
| bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_hongfan_grulstm_fgm |  0.5473 | 0.5497    |     |
| bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm__v3 | 0.5506  | 0.5584   | 
| bert120k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold4__v3 | 0.5648 | 0.56086   | 
|bert150k lr2e-5| | | |
| bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0__v3 | 0.5513 |    |     |
| bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_lr2e-5__v3 | 0.5552 | 0.5268   |    |
| bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_lr7e-5__v3 |  |    |    |
| bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_lr2e-5__v3(dice) | 0.5294 |    |  第二轮52第三轮就又飘了   |
| bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_fgm_ce__v3 | 0.5543 |    |     |
| bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_fgm__v3(dice) | 0.5494 | 0.5697   |     |
| bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_hongfan_grulstm_fold0_pgd__v3(multi_dice_fgm_pgd_2e-5) | 0.5645  | 0.5418   |     |
| bert150k_ce_supcon_0.1_0.5_sace_dr_msdrop_sum_4_0.4_multi_fold0_lr7e-5 | 0.5732  |    |     |
| train.bert150k_ce_supcon_lr1e-5_multi__fold0__v3   |               |               |
| train.bert150k_ce_supcon_lr1e-5_multi_fgm__fold0__v3   |               |               |
| train.bert150k_ce_supcon_lr1e-5_multi_pgd__fold0__v3   |   0.483     |               |
| train.bert150k_dice_supcon_lr1e-5__fold0__v3   |   0.5173      |               |
| train.bert150k_dice_supcon_lr1e-5_fgm__fold0__v3   |               |               |
| train.bert150k_dice_supcon_lr1e-5_multi_fgm__fold0__v3   |    0.5521      |               |
| train.bert150k_dice_supcon_lr1e-5_multi_pgd__fold0__v3   |      0.534       |               |

| train.bert300_dice_supcon_nolstmgru_nomulti_lr2e-5_nohong_warm200_nodr_nodorp__fold0__v3   |      0.5481       |               |
| train.bert300_dice_supcon_lr1e-5__fold0__v3   |      0.5587       |               |
| train.bert300_dice_supcon_nolstmgru_lr2e-5__fold0__v3   |  0.5429     |               |
| train.bert300_dice_supcon_nolstmgru_nomulti_lr2e-5__fold0__v3   | 0.5462       |               |
| train.bert300_dice_supcon_nolstmgru_lr2e-5_pgd__fold0__v3   | 0.5535/0.5669      |               |
| train.bert300_dice_supcon_lr1e-5_pgd__fold0__v3   |      0.5650      |  0.5660       |
| train.bert300_dice_supcon_lr1e-5_fgm__fold0__v3(lr2e-5)   |      0.5557       |               |
| train.bert300_dice_supcon__fold0__v3(supcon None lr2e-5)   |     0.5274     |               |

| 2train.bert300_0913_0   |     0.56     |               |
| 4train.newnezha_0914_0   |     0.5459  |               |

| train.newnezha_dice_supcon_lr1e-5_nomulti__fold0__v3(lr1e-5)   |     0.565  |               |
| train.newnezha_dice_supcon_lr1e-5_nomulti__fold0__v3(lr5e-5)   |     0.5635  |               |
| train.newnezha_dice_supcon_lr7e-5__fold0__v3   |     0.5639  |               |
| train.newnezha_dice_supcon_lr1e-5_nomulti_fgm__fold0__v3   |     0.5703  |  0.5636     |
| train.newnezha_dice_lr7e-5_fgm_drpooler__fold0__v3   |    0.5604   |               |
| final_ensemble/train.newnezha_dice_supcon_lr7e-5_fgm_drpooler__fold0__v3   |     0.5682  |               |
| train.newnezha_dice_supcon_lr1e-5_nomulti_pgd__fold0__v3(lr1e-5)   |     0.564  |               |
| train.newnezha_dice_supcon_lr1e-5_nomulti_pgd__fold0__v3(lr5e-5)   |    0.5663   |               |
             |

| train.newbert300_dice_supcon_lr1e-5__fold0__v3  |  0.5729  |   0.5627        |
| train.newbert300_dice_supcon_lr1e-5_57__fold1__v3  |  0.5514  |           |
| train.newbert300_dice_supcon_lr1e-5_57__fold2__v3  |  0.5527  |           |
| train.newbert300_dice_supcon_lr1e-5_57__fold3__v3  |  0.5581  |           |
| train.newbert300_dice_supcon_lr1e-5__fold0_drpooler__v3  |  0.5617  |          |
| train.newbert300_dice_supcon_915_0.575__v3  |  0.5542  |               |
| train.newbert300_dice_supcon_lr1e-5_fgm__fold0__v4  |  0.5585  |               |
| train.newbert300_dice_supcon_lr1e-5_pgd__fold0__v4  |  0.5662  |               |

| train.newbert300_dice_supcon_lr1e-5__fold2__v3  |  0.5518 |               |
| train.newbert300_dice_supcon_lr1e-5_pgd__fold0__v4  |  0.5659 |               |
| train.newnezha_dice_supcon_lr1e-5_nomulti_fgm_nodr_drop0.15__fold0__v3  |  0.5645 |               |

| train.bert150k_dice_supcon_multi_grulstm_fgm_lr2e-5_fold0__v3  |  0.537 |               |
| train.bert150k_dice_supcon_grulstm_fgm_lr2e-5_fold0__v3  |  0.5437 |               |
| train.bert150k_ce_supcon_grulstm_lr7e-5_fold0__v3  | 0.5464  |               |
| train.bert150k_dice_supcon_grulstm_pgd_lr1e-5_fold0__v3  | 0.5514  |               |

| train.newnezha500_dice_supcon_lr1e-5_dr__fold0__v3  | 0.5598  |               |
| train.newnezha500_dice_supcon_lr1e-5__fold0__v3  | 0.5598  |               |
| train.newnezha500_dice_supcon_lr1e-5_fgm__fold0__v3  | 0.5639  |               |
| train.newnezha500_dice_supcon_lr1e-5_pgd__fold0__v3  | 0.5687  |               |
| train.newnezha500_dice_supcon_lr1e-5_pgd__fold1__v3  |   |               |

| train.newbert300_dice_supcon_lr1e-5_pgd__fold1_plabel__v3  | 0.5846 |               |
| train.nezha120000_dice_supcon_lr1e-5_pgd__fold0__v3  | 0.5684 |               |



|新四折| | | |
| train.newnezha_dice_supcon_lr1e-5_nomulti_fgm__fold0__v3  |    0.5654   |               |
| train.newnezha_dice_supcon_lr1e-5_nomulti_fgm__fold1__v3  |    0.5668   |               |
| train.newnezha_dice_supcon_lr1e-5_nomulti_fgm__fold2__v3  |    0.5675   |               |
| train.newnezha_dice_supcon_lr1e-5_nomulti_fgm__fold3__v3  |    0.5597   |  
| train.newbert300_dice_supcon_lr1e-5_pgd__fold0__v3  |  0.5659  |               |
| train.newbert300_dice_supcon_lr1e-5_pgd__fold1__v3  |  0.573   | 0.5857      |
| train.newbert300_dice_supcon_lr1e-5_pgd__fold2__v3  |  0.5561  |               |
| train.newbert300_dice_supcon_lr1e-5_pgd__fold3__v3  |  0.5678 |               |
| train.bert150k_dice_supcon_grulstm_fgm_lr1e-5_fold0__v3  | 0.5591/0.5437  |               |
| train.bert150k_dice_supcon_grulstm_fgm_lr1e-5_fold1__v3  | 0.5635  |               |
| train.bert150k_dice_supcon_grulstm_fgm_lr1e-5_fold2__v3  | 0.563  |               |
| train.bert150k_dice_supcon_grulstm_fgm_lr1e-5_fold3__v3  | 0.533  |               |
| train.nezha110000_dice_supcon_lr1e-5_pgd__fold0__v3  | 0.5701  |               |
| train.nezha110000_dice_supcon_lr1e-5_pgd__fold1__v3  | 0.5646  |               |
| train.nezha110000_dice_supcon_lr1e-5_pgd__fold2__v3  | 0.5632  |               |
| train.nezha110000_dice_supcon_lr1e-5_pgd__fold3__v3  | 0.5768  |               |
