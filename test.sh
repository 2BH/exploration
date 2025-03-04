task=MultiRoomLava-N4-S5
# steps=10000000
int_rew=DEIR
model_algo=DEIR_end
video=0
for model_steps in  480000 960000 1440000 1920000 2400000 4800000 #48000 96000 144000 192000 240000 288000 336000 384000 432000 
do
# path=/home/yan/work_spaces/exploration/logs/MiniGrid-MultiRoomLava-N6-v0/2024-09-28_12-11-44/26192416/rl_model_9600000_steps.zip
# path=/home/yan/work_spaces/exploration/logs/MultiRoomLava-N4-S5-v0/2024-09-25_16-48-31/72346654/rl_model_9600000_steps.zip
# path=/home/yan/work_spaces/exploration/logs/MiniGrid-DoorKeyLava-8x8-v0/test/rl_model_4800000_steps.zip
root_dir=/home/yan/Documents/paper/paper_yan25iclr/experiment/minigrid/model/MRL4_new/
path="${root_dir}${model_algo}/${model_algo}_rl_model_${model_steps}_steps.zip"
# echo $path
for seed in 26202127 26192416 1484620 72346654 32225970 31415816 68630553 42161619 14201156 30132438
do 
    PYTHONPATH=./ python3 src/test.py --int_rew_source=$int_rew --run_id=$seed --game_name=$task \
    --model_path=$path --record_video=$video --n_eval_episodes=1

done
# python action_count.py
done