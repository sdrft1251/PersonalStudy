# !/bin/bash

# 파일 조회
function read_file_list(){
    local searh_dir=$1
    for entry in $searh_dir/*
    do
        if [ "${entry:(-5)}" == ".json" ]; then
            echo ${entry##*/}
        fi
    done
}

# To do list
echo "To Do list"
read_file_list /works/ai2/train_manager/to_do
# Doing List
echo "Doing list"
read_file_list /works/ai2/train_manager/doing
# Done List
echo "Done list"
read_file_list /works/ai2/train_manager/done

echo "------------------------------------"
my_slack_webhook=""

# To Do 에서 파일 읽어 오기
searchdir=/works/ai2/train_manager/to_do
for entry in $searchdir/*
do
    if [ "${entry:(-5)}" == ".json" ]; then
        echo "Start ${entry##*/}"
        # Json 파일 이름들 가져오기
        jsonfilename=${entry##*/}
        # 진행 중인 파일 Doing으로 이동
        mv /works/ai2/train_manager/to_do/$jsonfilename /works/ai2/train_manager/doing/
        # 훈련 시작
        python train_loop.py ./doing/$jsonfilename
        # 정상 출력 확인
        if [ $? -eq 0 ];then
            echo "Train Done"
            mv /works/ai2/train_manager/doing/$jsonfilename /works/ai2/train_manager/done/
            # Slack으로 메시지
            json="{\"text\": \"Training $jsonfilename is done\"}"
            curl -X POST -H 'Content-type: application/json' --data "$json" $my_slack_webhook
        # 비정상으로 종료 되었을 때,
        else
            echo "Train Fail!"
            mv /works/ai2/train_manager/doing/$jsonfilename /works/ai2/train_manager/failed/
            # Slack에 메시지
            json="{\"text\": \"Training $jsonfilename is Failed\"}"
            curl -X POST -H 'Content-type: application/json' --data "$json" $my_slack_webhook
        fi
    fi
done

endMassage="{\"text\": \"All Process is done\"}"
curl -X POST -H 'Content-type: application/json' --data "$endMassage" $my_slack_webhook