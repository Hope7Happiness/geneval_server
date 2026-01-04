# # 设置 CUDA 根目录为你的 conda 环境路径
# export CUDA_HOME=/data/scratch-oc40/zhh24/anaconda3/envs/geneval

# # 确保 nvcc 在路径中（你已经有了，但重新确认一下无妨）
# export PATH=$CUDA_HOME/bin:$PATH

# # 确保链接库能被找到
# export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# 确保目标目录存在
mkdir -p /data/scratch-oc40/zhh24/anaconda3/envs/geneval/include

# 创建软链接，将 targets 下的 include 内容链接到环境根目录的 include
ln -s /data/scratch-oc40/zhh24/anaconda3/envs/geneval/targets/x86_64-linux/include/* /data/scratch-oc40/zhh24/anaconda3/envs/geneval/include/