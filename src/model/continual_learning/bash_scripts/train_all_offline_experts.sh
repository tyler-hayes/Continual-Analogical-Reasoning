### OFFLINE MODELS WITH CS TASK FIRST

export NAME=cs_6
T=('cs' 'io' 'lr' 'ud' 'd4' 'd9' '4c')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=cs_5
T=('cs' 'io' 'lr' 'ud' 'd4' 'd9')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=cs_4
T=('cs' 'io' 'lr' 'ud' 'd4')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=cs_3
T=('cs' 'io' 'lr' 'ud')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=cs_2
T=('cs' 'io' 'lr')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=cs_1
T=('cs' 'io')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

### OFFLINE MODELS WITH UD TASK FIRST

export NAME=ud_6
T=('ud' 'cs' 'io' '4c' 'd9' 'd4' 'lr')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=ud_5
T=('ud' 'cs' 'io' '4c' 'd9' 'd4')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=ud_4
T=('ud' 'cs' 'io' '4c' 'd9')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=ud_3
T=('ud' 'cs' 'io' '4c')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=ud_2
T=('ud' 'cs' 'io')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=ud_1
T=('ud' 'cs')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

### OFFLINE MODELS WITH D4 TASK FIRST

export NAME=d4_6
T=('d4' 'lr' '4c' 'ud' 'd9' 'cs' 'io')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=d4_5
T=('d4' 'lr' '4c' 'ud' 'd9' 'cs')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=d4_4
T=('d4' 'lr' '4c' 'ud' 'd9')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=d4_3
T=('d4' 'lr' '4c' 'ud')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=d4_2
T=('d4' 'lr' '4c')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh

export NAME=d4_1
T=('d4' 'lr')
export TASK="${T[@]}"
chmod +x offline_experts.sh
./offline_experts.sh


