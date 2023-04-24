from flask import request, jsonify

from . import dbtest
from viewer import db
from viewer.model import EcgTest, TestGroup, SampleGroup, PreprocessGroup, TestLink, SampleLink

#################################################
#### 테스트용 데이터베이스에 데이터 추가 & View 를 위한 ####
#################################################

#### For Post ####

@dbtest.route('/dbtest/ecgtest', methods=['POST'])
def ecgtests_post():
    # ===== arg =====
    region = request.args.get('region', type=str)
    test_id = request.args.get('test_id', type=str)
    duration = request.args.get('duration', type=str)
    condition = request.args.get('is_sample', type=str)
    edf_path = request.args.get('edf_path', type=str)
    details_path = request.args.get('details_path', type=str)

    ect_test = EcgTest(region=region, test_id=test_id, duration=duration, condition=condition,\
    edf_path=edf_path, details_path=details_path)

    db.session.add(ect_test)
    db.session.commit()

    return f"{region}_{test_id} <- Add Success"


@dbtest.route('/dbtest/testgroup', methods=['POST'])
def testgroup_post():
    # ===== arg =====
    group_name = request.args.get('group_name', type=str)
    group_status = request.args.get('group_status', type=str)

    test_group = TestGroup(group_name=group_name, group_status=group_status)
    db.session.add(test_group)
    db.session.commit()

    return f"{group_name} <- Add Success"


@dbtest.route('/dbtest/samplegroup', methods=['POST'])
def samplegroup_post():
    # ===== arg =====
    group_name = request.args.get('group_name', type=str)
    group_status = request.args.get('group_status', type=str)

    sample_group = SampleGroup(group_name=group_name, group_status=group_status)
    db.session.add(sample_group)
    db.session.commit()

    return f"{group_name} <- Add Success"


@dbtest.route('/dbtest/preprocessgroup', methods=['POST'])
def preprocessgroup_post():
    # ===== arg =====
    group_name = request.args.get('group_name', type=str)
    group_status = request.args.get('group_status', type=str)
    path = request.args.get('path', type=str)

    preprocess_group = PreprocessGroup(group_name=group_name, group_status=group_status, path=path)
    db.session.add(preprocess_group)
    db.session.commit()

    return f"{group_name} <- Add Success"



#### For View ####

@dbtest.route('/dbtest/testgroup/<string:ty>', methods=['GET'])
def testgroup_get(ty):   # ty = "t" | "s" | "p"
    model_obj = None
    if ty == "t":
        model_obj = TestGroup
    elif ty == "s":
        model_obj = SampleGroup
    else:
        model_obj = PreprocessGroup

    group_list = model_obj.query.all()
    dump = []
    for group_ in group_list:
        dump.append({
            "id": group_.id,
            "group_name": group_.group_name,
        })
    return jsonify({"All-Data":dump})


@dbtest.route('/dbtest/link/<string:ty>', methods=['GET'])
def linkview(ty):   # ty = "t" | "s"
    dump = []
    if ty == "t":
        link_all = TestLink.query.all()
        for link in link_all:
                dump.append({
                "ecgtest_id": link.ecgtest_id,
                "testgroup_id": link.testgroup_id,
            })
    else:
        link_all = SampleLink.query.all()
        for link in link_all:
                dump.append({
                "ecgtest_id": link.ecgtest_id,
                "samplegroup_id": link.samplegroup_id,
            })
    
    return jsonify({"All-Data":dump})
