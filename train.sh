task=DoorKeyLava-8x8
steps=10000000
int_rew=DEIR
for seed in 26202127 #26192416 1484620 72346654 32225970 31415816 68630553 42161619 14201156 30132438
do 
    PYTHONPATH=./ python3 src/train.py --int_rew_source=$int_rew --group_name=gage --project_name=gage --run_id=$seed --game_name=$task --total_steps=$steps \
    --optim_reward=0.8 --gage_topk_init=6 --gage_eta1=1.5 --gage_eta2=2
    PYTHONPATH=./ python3 src/train.py --int_rew_source=$int_rew --group_name=gage --project_name=gage --run_id=$seed --game_name=$task --total_steps=$steps \
    --optim_reward=0.8 --gage_topk_init=6 --gage_eta1=1.9 --gage_eta2=2
    PYTHONPATH=./ python3 src/train.py --int_rew_source=$int_rew --group_name=gage --project_name=gage --run_id=$seed --game_name=$task --total_steps=$steps \
    --optim_reward=0.8 #--gage_topk_init=6 --gage_eta1=1.5 --gage_eta2=2
    PYTHONPATH=./ python3 src/train.py --int_rew_source=$int_rew --group_name=gage --project_name=gage --run_id=$seed --game_name=$task --total_steps=$steps \
    --optim_reward=0.8 --gage_topk_init=7 --gage_eta1=1.3 --gage_eta2=2
    PYTHONPATH=./ python3 src/train.py --int_rew_source=$int_rew --group_name=gage --project_name=gage --run_id=$seed --game_name=$task --total_steps=$steps \
    --optim_reward=0.8 --gage_topk_init=6 --gage_eta1=1.7 --gage_eta2=2
    PYTHONPATH=./ python3 src/train.py --int_rew_source=$int_rew --group_name=gage --project_name=gage --run_id=$seed --game_name=$task --total_steps=$steps \
    --optim_reward=0.8 --gage_topk_init=7 --gage_eta1=1.7 --gage_eta2=2
done