# for seed in 26202127 #26192416 1484620 72346654 32225970 31415816 68630553 42161619 14201156 30132438
# do PYTHONPATH=./ python3 src/train.py --int_rew_source=DEIR --group_name=gage --project_name=gage\
#     --run_id=$seed --game_name=FourRooms-Lava #--gage_topk_init= --gage_eta1= --gage_eta2=
# done
for seed in 26202127 #26192416 1484620 72346654 32225970 31415816 68630553 42161619 14201156 30132438
do PYTHONPATH=./ python3 src/train.py --int_rew_source=DEIR --project_name=gage\
    --run_id=$seed --game_name=FourRooms-Lava #--gage_topk_init= --gage_eta1= --gage_eta2=
done