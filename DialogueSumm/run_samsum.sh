cuda=0
model_file=BiDeN
max_length=800
summary_max_length=100
lr=2e-5
batch_size=4
epochs=10
save_path=save
model_name=facebook/bart-large

python3 myTrain.py \
    --cuda $cuda \
    --model_file $model_file \
    --max_length $max_length \
    --learning_rate $lr \
    --batch_size $batch_size \
    --epochs $epochs \
    --model_name $model_name \
    --summary_max_length $summary_max_length \
    --dataset samsum \
    --save_path $save_path
