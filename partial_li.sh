#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=er33
#SBATCH -o out33 -e err33
#SBATCH -N 1 -n 6 -t 240:00:00
#SBATCH --mail-user=liuqiong_yl@outllok.com
#SBATCH --gres=gpu:1 
#SBATCH -p gnall

# python SAC_OSTC_sa_partial_1.py --partial 0  --env_name 69 &


# python SAC_OSTC_sa_partial_1.py --partial 11  --env_name 69 & 
# python SAC_OSTC_sa_partial_1.py --partial 12  --env_name 69 &
# python SAC_OSTC_sa_partial_1.py --partial 15  --env_name 69 &
# python SAC_OSTC_sa_partial_1.py --partial 16  --env_name 69 &

# python SAC_OSTC_sa_partial_1.py --partial 17  --env_name 69 & 
# python SAC_OSTC_sa_partial_1.py --partial 18  --env_name 69 &
# python SAC_OSTC_sa_partial_1.py --partial 19  --env_name 69 &
# python SAC_OSTC_sa_partial_1.py --partial 20  --env_name 69 &

# python SAC_OSTC_sa_partial_1.py --partial 21  --env_name 69 & 
# python SAC_OSTC_sa_partial_1.py --partial 22  --env_name 69 &
# python SAC_OSTC_sa_partial_1.py --partial 23  --env_name 69 &
# python SAC_OSTC_sa_partial_1.py --partial 24  --env_name 69 &

# python SAC_OSTC_sa_partial_1.py --partial 0  --env_name 118 &


# python SAC_OSTC_sa_partial_1.py --partial 11  --env_name 118 & 
# python SAC_OSTC_sa_partial_1.py --partial 12  --env_name 118 &
# python SAC_OSTC_sa_partial_1.py --partial 15  --env_name 118 &
# python SAC_OSTC_sa_partial_1.py --partial 16  --env_name 118 &

# python SAC_OSTC_sa_partial_1.py --partial 17  --env_name 118 & 
# python SAC_OSTC_sa_partial_1.py --partial 18  --env_name 118 &
# python SAC_OSTC_sa_partial_1.py --partial 19  --env_name 118 &
# python SAC_OSTC_sa_partial_1.py --partial 20  --env_name 118 &

# python SAC_OSTC_sa_partial_1.py --partial 21  --env_name 118 & 
# python SAC_OSTC_sa_partial_1.py --partial 22  --env_name 118 &
# python SAC_OSTC_sa_partial_1.py --partial 23  --env_name 118 &
# python SAC_OSTC_sa_partial_1.py --partial 24  --env_name 118 &


python SAC_OSTC_sa_partial_1.py --partial 11  --env_name 33 & 
python SAC_OSTC_sa_partial_1.py --partial 12  --env_name 33 &
python SAC_OSTC_sa_partial_1.py --partial 15  --env_name 33 &
python SAC_OSTC_sa_partial_1.py --partial 16  --env_name 33 &

python SAC_OSTC_sa_partial_1.py --partial 17  --env_name 33 & 
python SAC_OSTC_sa_partial_1.py --partial 18  --env_name 33 &
python SAC_OSTC_sa_partial_1.py --partial 19  --env_name 33 &
python SAC_OSTC_sa_partial_1.py --partial 20  --env_name 33 &

python SAC_OSTC_sa_partial_1.py --partial 21  --env_name 33 & 
python SAC_OSTC_sa_partial_1.py --partial 22  --env_name 33 &
python SAC_OSTC_sa_partial_1.py --partial 23  --env_name 33 &
python SAC_OSTC_sa_partial_1.py --partial 24  --env_name 33 &

wait
