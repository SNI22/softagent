xhost +local:root     

  docker run \
    --rm \
    --runtime=nvidia \
    --network=host \
    --gpus all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /dev:/dev \
    -v /home/sni22/miniconda3:/home/sni22/miniconda3 \
    -v /home/sni22/Documents/softagent:/home/sni22/softagent \
    -it xingyu/softgym:latest bash


export PATH=/home/sni22/miniconda3/bin:$PATH
source /home/sni22/miniconda3/etc/profile.d/conda.sh
cd /home/sni22/softagent/softgym
conda env create -f environment.yml
conda activate softgym
. ./prepare_1.0.sh && ./compile_1.0.sh
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python examples/random_env.py --env_name PassWater
cd ..
. ./prepare_1.0.sh