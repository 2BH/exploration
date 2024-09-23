for seed in 26202127 #26192416 1484620 72346654 32225970 31415816 68630553 42161619 14201156 30132438
do 
    PYTHONPATH=./ python3 src/train.py --int_rew_source=DEIR --group_name=gage --project_name=gage\
    --run_id=$seed --game_name=FourRooms-Lava --optim_reward=0.8 #--gage_topk_init=9 --gage_eta1=1.0 --gage_eta2=2
done