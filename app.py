from flask import Flask, render_template, redirect, url_for, session, request, jsonify
import mysql.connector
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import math
import sqlite3

from itertools import combinations

app = Flask(__name__)
app.secret_key = 'super_secret_key'


color_mapping2 = {(231, 47, 39): ('R', 'V'),
(207, 46, 49): ('R', 'S'),
(231, 108, 86): ('R', 'B'),
(233, 163, 144): ('R', 'P'),
(236, 217, 202): ('R', 'Vp'),
(213, 182, 166): ('R', 'Lgr'),
(211, 142, 110): ('R', 'L'),
(171, 131, 115): ('R', 'Gr'),
(162, 88, 61): ('R', 'Dl'),
(172, 36, 48): ('R', 'Dp'),
(116, 47, 50): ('R', 'Dk'),
(79, 46, 43): ('R', 'Dgr'),
(238, 113, 25): ('YR', 'V'),
(226, 132, 45): ('YR', 'S'),
(241, 176, 102): ('YR', 'B'),
(242, 178, 103): ('YR', 'P'),
(235, 223, 181): ('YR', 'Vp'),
(218, 196, 148): ('YR', 'Lgr'),
(215, 145, 96): ('YR', 'L'),
(158, 128, 110): ('YR', 'Gr'),
(167, 100, 67): ('YR', 'Dl'),
(169, 87, 49): ('YR', 'Dp'),
(115, 63, 44): ('YR', 'Dk'),
(85, 55, 43): ('YR', 'Dgr'),
(255, 200, 8): ('Y', 'V'),
(227, 189, 28): ('Y', 'S'),
(255, 228, 15): ('Y', 'B'),
(255, 236, 79): ('Y', 'P'),
(249, 239, 189): ('Y', 'Vp'),
(233, 227, 143): ('Y', 'Lgr'),
(255, 203, 88): ('Y', 'L'),
(148, 133, 105): ('Y', 'Gr'),
(139, 117, 65): ('Y', 'Dl'),
(156, 137, 37): ('Y', 'Dp'),
(103, 91, 44): ('Y', 'Dk'),
(75, 63, 45): ('Y', 'Dgr'),
(170, 198, 27): ('GY', 'V'),
(162, 179, 36): ('GY', 'S'),
(169, 199, 35): ('GY', 'B'),
(219, 220, 93): ('GY', 'P'),
(228, 235, 191): ('GY', 'Vp'),
(209, 116, 73): ('GY', 'Lgr'),
(195, 202, 101): ('GY', 'L'),
(144, 135, 96): ('GY', 'Gr'),
(109, 116, 73): ('GY', 'Dl'),
(91, 132, 47): ('GY', 'Dp'),
(54, 88, 48): ('GY', 'Dk'),
(44, 60, 49): ('GY', 'Dgr'),
(19, 166, 50): ('G', 'V'),
(18, 154, 47): ('G', 'S'),
(88, 171, 45): ('G', 'B'),
(155, 196, 113): ('G', 'P'),
(221, 232, 207): ('G', 'Vp'),
(179, 202, 157): ('G', 'Lgr'),
(141, 188, 90): ('G', 'L'),
(143, 162, 121): ('G', 'Gr'),
(88, 126, 61): ('G', 'Dl'),
(20, 114, 48): ('G', 'Dp'),
(30, 98, 50): ('G', 'Dk'),
(34, 62, 51): ('G', 'Dgr'),
(4, 148, 87): ('BG', 'V'),
(6, 134, 84): ('BG', 'S'),
(43, 151, 89): ('BG', 'B'),
(146, 198, 131): ('BG', 'P'),
(209, 234, 211): ('BG', 'Vp'),
(166, 201, 163): ('BG', 'Lgr'),
(140, 195, 110): ('BG', 'L'),
(122, 165, 123): ('BG', 'Gr'),
(39, 122, 62): ('BG', 'Dl'),
(23, 106, 43): ('BG', 'Dp'),
(27, 86, 49): ('BG', 'Dk'),
(31, 56, 45): ('BG', 'Dgr'),
(1, 134, 141): ('B', 'V'),
(3, 130, 122): ('B', 'S'),
(0, 147, 159): ('B', 'B'),
(126, 188, 209): ('B', 'P'),
(194, 222, 242): ('B', 'Vp'),
(127, 175, 166): ('B', 'Lgr'),
(117, 173, 169): ('B', 'L'),
(130, 154, 145): ('B', 'Gr'),
(24, 89, 63): ('B', 'Dl'),
(20, 88, 60): ('B', 'Dp'),
(18, 83, 65): ('B', 'Dk'),
(29, 60, 47): ('B', 'Dgr'),
(3, 86, 155): ('PB', 'V'),
(6, 113, 148): ('PB', 'S'),
(59, 130, 157): ('PB', 'B'),
(147, 184, 213): ('PB', 'P'),
(203, 215, 232): ('PB', 'Vp'),
(165, 184, 199): ('PB', 'Lgr'),
(138, 166, 187): ('PB', 'L'),
(133, 154, 153): ('PB', 'Gr'),
(53, 109, 98): ('PB', 'Dl'),
(8, 87, 107): ('PB', 'Dp'),
(16, 76, 84): ('PB', 'Dk'),
(25, 62, 63): ('PB', 'Dgr'),
(46, 20, 141): ('P', 'V'),
(92, 104, 163): ('P', 'S'),
(178, 137, 166): ('P', 'B'),
(197, 188, 213): ('P', 'P'),
(224, 218, 230): ('P', 'Vp'),
(184, 190, 189): ('P', 'Lgr'),
(170, 165, 199): ('P', 'L'),
(151, 150, 139): ('P', 'Gr'),
(44, 77, 143): ('P', 'Dl'),
(58, 55, 119): ('P', 'Dp'),
(40, 57, 103): ('P', 'Dk'),
(34, 54, 68): ('P', 'Dgr'),
(204, 63, 92): ('RP', 'V'),
(175, 92, 87): ('RP', 'S'),
(209, 100, 109): ('RP', 'B'),
(218, 176, 176): ('RP', 'P'),
(235, 219, 224): ('RP', 'Vp'),
(206, 185, 179): ('RP', 'Lgr'),
(205, 154, 149): ('RP', 'L'),
(160, 147, 131): ('RP', 'Gr'),
(115, 71, 79): ('RP', 'Dl'),
(111, 61, 56): ('RP', 'Dp'),
(88, 60, 50): ('RP', 'Dk'),
(53, 52, 48): ('RP', 'Dgr'),
(244, 244, 244): ('N', '9.5'),
(236, 236, 236): ('N', '9'),
(206, 206, 206): ('N', '8'),
(180, 180, 180): ('N', '7'),
(152, 152, 152): ('N', '6'),
(126, 126, 126): ('N', '5'),
(86, 86, 86): ('N', '4'),
(60, 60, 60): ('N', '3'),
(38, 38, 38): ('N', '2'),
(10, 10, 10): ('N', '1.5')}

tuple_list=[[('Y', 'P'), ('R', 'P'), ('YR', 'Vp')], [('RP', 'P'), ('Y', 'Vp'), ('BG', 'Lgr')], [('RP', 'B'), ('Y', 'Vp'), ('PB', 'B')], [('Y', 'B'), ('R', 'B'), ('P', 'B')], [('RP', 'B'), ('BG', 'B'), ('P', 'V')], [('RP', 'V'), ('Y', 'V'), ('P', 'Dp')], [('Y', 'B'), ('BG', 'V'), ('N', '1.5')], [('BG', 'S'), ('Y', 'P'), ('PB', 'Dk')], [('R', 'V'), ('N', '1.5'), ('B', 'V')], [('YR', 'Dk'), ('R', 'S'), ('P', 'Dk')], [('RP', 'Dp'), ('Y', 'Dl'), ('P', 'Dk')], [('YR', 'Dk'), ('R', 'Dl'), ('RP', 'Dk')], [('R', 'Dp'), ('P', 'Dk'), ('BG', 'Dl')], [('Y', 'Dl'), ('R', 'Dk'), ('P', 'S')], [('RP', 'Dp'), ('GY', 'Dl'), ('P', 'Dk')], [('N', '1.5'), ('Y', 'Gr'), ('B', 'Dk')], [('Y', 'Dgr'), ('Y', 'Gr'), ('PB', 'Dk')], [('B', 'Dk'), ('Y', 'Gr'), ('N', '1.5')], [('N', '1.5'), ('N', '7'), ('BG', 'Dk')], [('P', 'Dgr'), ('N', '9'), ('B', 'Dp')], [('P', 'Dgr'), ('N', '9'), ('N', '5')], [('N', '7'), ('YR', 'Gr'), ('N', '4')], [('BG', 'Gr'), ('N', '9'), ('N', '5')], [('YR', 'Gr'), ('N', '8'), ('PB', 'Gr')], [('R', 'Gr'), ('N', '8'), ('GY', 'Gr')], [('YR', 'Gr'), ('Y', 'Lgr'), ('Y', 'Gr')], [('Y', 'Vp'), ('GY', 'S'), ('Y', 'Dl')], [('Y', 'Lgr'), ('G', 'Vp'), ('G', 'Lgr')], [('GY', 'S'), ('N', '8'), ('G', 'Gr')], [('YR', 'Vp'), ('GY', 'Lgr'), ('N', '7')], [('R', 'Vp'), ('RP', 'P'), ('P', 'Vp')], [('R', 'Lgr'), ('N', '9'), ('GY', 'Lgr')], [('YR', 'Vp'), ('BG', 'Vp'), ('GY', 'Vp')], [('G', 'Vp'), ('N', '9.5'), ('B', 'P')], [('BG', 'P'), ('N', '9.5'), ('B', 'Vp')], [('G', 'Vp'), ('N', '9.5'), ('B', 'P')], [('Y', 'B'), ('N', '9.5'), ('GY', 'B')], [('PB', 'B'), ('GY', 'P'), ('B', 'B')], [('GY', 'V'), ('Y', 'P'), ('G', 'S')]]

# emotion = ["pretty","colorful","active","mellow","elaborate","Dapper","metallic","chic","calm","tranquil","soft","clear","youthful"]
emotion = ["화려한", "다채로운", "활기찬", "부드러운", "정교한", "멋진", "세련된","시크한","차분한","고요한","부드러운","깔끔한","에너지"]
db_config = {
    'host': 'restartdb.c588s0060coo.ap-northeast-2.rds.amazonaws.com',
    'user': 'admin',
    'password': 'restartpwd',
    'database': 'restartdb'
}
@app.route('/')
def mainpage():
    return render_template('mainpage.html')



@app.route('/gallery', methods=['GET', 'POST'])
def gallery():
    db = mysql.connector.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )
    cursor = db.cursor(dictionary=True)

    if request.method == 'POST':
        selected_emotions = request.form.getlist('emotion[]')
        session['selected_emotions'] = selected_emotions  # 사용자가 선택한 감정 값을 세션에 저장

        if selected_emotions:
            conditions = []
            for emotion in selected_emotions:
                condition = f"(%s IN (emotion1, emotion2, emotion3))"
                conditions.append(condition)

            query = "SELECT * FROM images WHERE " + " AND ".join(conditions)
            query_parameters = tuple(selected_emotions)
            cursor.execute(query, query_parameters)
        else:
            query = "SELECT * FROM images"
            cursor.execute(query)
        images = cursor.fetchall()
    else:
        selected_emotions = session.get('selected_emotions', [])  # 세션에서 감정 값을 불러옴
        if selected_emotions:
            conditions = []
            for emotion in selected_emotions:
                condition = f"(%s IN (emotion1, emotion2, emotion3))"
                conditions.append(condition)

            query = "SELECT * FROM images WHERE " + " AND ".join(conditions)
            query_parameters = tuple(selected_emotions)
            cursor.execute(query, query_parameters)
            images = cursor.fetchall()
        else:
            query = "SELECT * FROM images"
            cursor.execute(query)
            images = cursor.fetchall()

    for image in images:
        # 경로에서 프로젝트 경로 부분을 제거하고, 모든 '\\'를 '/'로 교체합니다.
        relative_path = image['image_path'].replace('\\', '/').replace(
            'C:/Users/yunho/PycharmProjects/RestArt/static/',
            '')
        image['image_path'] = relative_path

    db.close()
    return render_template('gallery.html', images=images, selected_emotions=selected_emotions)



@app.route('/artistInform')
def artistInform():
    return render_template('artistInform.html')

@app.route('/signpage')
def signpage():
    return render_template('signpage.html')

@app.route('/artenroll')
def artenroll():
    return render_template('artenroll.html')

@app.route('/exhibition')
def exhibition():
    return render_template('exhibition.html')

@app.route('/login')
def login():
    return render_template('login.html')

# @app.route('/recomm')
# def recomm():
#     return render_template('recomm.html')

@app.route('/recomm')
def recomm():
    uploaded_image = session.get('uploaded_image', None)
    fin = session.get('fin', ['값1', '값2', '값3'])

    db = mysql.connector.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )
    cursor = db.cursor(dictionary=True)

    conditions = []
    for emotion in fin:
        condition = f"('{emotion}' IN (emotion1, emotion2, emotion3))"
        conditions.append(condition)

    query = "SELECT * FROM images WHERE " + " AND ".join(conditions)
    cursor.execute(query)
    matched_images = cursor.fetchall()

    for image in matched_images:
        # 이미지 경로에서 파일 이름 추출
        file_name = image['image_path'].split('\\')[-1]

        # 언더바로 분리 가능한 경우에만 분리
        if '_' in file_name:
            artist_name, artwork_name = file_name.split('_', 1)
            artwork_name = artwork_name.rsplit('.', 1)[0]  # 확장자 제거
        else:
            # 언더바가 없는 경우, 파일 이름 전체를 작품명으로 사용
            artist_name = "알 수 없음"  # 작가명을 알 수 없음으로 처리
            artwork_name = file_name.rsplit('.', 1)[0]  # 확장자 제거

        image['artist_name'] = artist_name
        image['artwork_name'] = artwork_name
        # 이미지 경로 조정
        image['image_path'] = image['image_path'].replace('\\', '/').replace(
            'C:/Users/yunho/PycharmProjects/RestArt/static/', '')
    db.close()
    return render_template('recomm.html', uploaded_image=uploaded_image, fin=fin, matched_images=matched_images)


def save_emotions_to_database(image_path, emotions, db_config):
    # 데이터베이스 연결 설정
    db = mysql.connector.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database']
    )
    cursor = db.cursor()

    # 이미지 경로와 감정 데이터를 데이터베이스에 저장
    query = "INSERT INTO images (image_path, emotion1, emotion2, emotion3) VALUES (%s, %s, %s, %s)"
    values = (image_path, emotions[0], emotions[1], emotions[2])

    cursor.execute(query, values)
    db.commit()
    cursor.close()
    db.close()


def extract_colors_and_graph(image_path, num_colors=10):
    try:
        # 이미지 로드 및 리사이즈
        image = imread(image_path)
        image = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)

        # 이미지 데이터를 2D 배열로 변환
        pixels = np.reshape(image, (image.shape[0] * image.shape[1], 3))

        # KMeans 클러스터링
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)

        # 클러스터 중심 색상 추출
        colors1 = kmeans.cluster_centers_ * 255  # 정규화된 값을 원래 범위로 되돌림
        colors = kmeans.cluster_centers_

        # RGB 값을 문자열로 변환하여 레이블 생성
        color_labels = [f'({int(color[0])}, {int(color[1])}, {int(color[2])})' for color in colors1]

        # 각 클러스터의 비율 계산
        counts = np.bincount(kmeans.labels_)
        max_count = np.sum(counts)
        percentage = (counts / max_count) * 100  # 퍼센트로 변환

        # 클러스터링된 이미지 재구성 및 저장
        clustered_pixels = np.array([colors[label] for label in kmeans.labels_])
        clustered_image = np.reshape(clustered_pixels, (image.shape[0], image.shape[1], 3))
        clustered_image_path = 'static/uploads/clustered_image.png'
        plt.imsave(clustered_image_path, clustered_image)

        # 세로 막대 그래프 생성
        plt.figure(figsize=(6, 4))
        bars = plt.bar(range(num_colors), percentage, color=colors)

        # 상위 3개 클러스터에 별 표시 추가
        top_indices = np.argsort(percentage)[-3:]  # 상위 3개 인덱스
        for idx in top_indices:
            plt.text(bars[idx].get_x() + bars[idx].get_width() / 2, bars[idx].get_height(), '*', ha='center', va='bottom', color='gold', fontsize=20)

        plt.xlabel('Clusters')
        plt.ylabel('Percentage (%)')
        plt.title('Color Clusters')
        plt.xticks(range(num_colors), color_labels, rotation=90)

        # 그래프 이미지 저장
        graph_image_path = 'static/uploads/color_bar_graph.png'
        plt.savefig(graph_image_path, bbox_inches='tight')
        counts = np.bincount(kmeans.labels_)
        max_count = np.sum(counts)
        percentage = (counts / max_count) * 100
        cluster_centers = kmeans.cluster_centers_
        cluster_centers_255 = cluster_centers * 255
        list_RGB = []
        cnt = 0
        # 클러스터 별 RGB 값(0~255 범위)과 비율을 함께 출력
        print("\n클러스터 별 RGB 값(0~255)과 비율:")
        for i, (cluster_center, count) in enumerate(zip(cluster_centers_255, counts)):
            ratio = count / pixels.shape[0]  # 전체 픽셀 대비 현재 클러스터 픽셀 비율
            list_RGB.append([cluster_center.round(0).astype(int), ratio.round(2)])
            if cnt == 3:
                break

            cnt += 1

        sorted_list_RGB = sorted(list_RGB, key=lambda x: x[1], reverse=True)

        # 상위 3개의 요소 선택 (첫 번째 요소는 가장 큰 비율을 가진 요소를 포함)
        top_combinations = [sorted_list_RGB[0], sorted_list_RGB[1], sorted_list_RGB[2]]
        new_list_RGB2 = [item[0] for item in sorted_list_RGB]

        new_list_RGB = [[new_list_RGB2[0],new_list_RGB2[1],new_list_RGB2[2]],
                        [new_list_RGB2[0],new_list_RGB2[1],new_list_RGB2[3]],
                        [new_list_RGB2[0],new_list_RGB2[2],new_list_RGB2[3]]]

        # 그냥 리스트로 받아 오기
        new_mapping = []

        for value in tuple_list:
            for item in value:
                if item in color_mapping2.values():
                    key = [k for k, v in color_mapping2.items() if v == item][0]
                    new_mapping.append(key)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    def euclidean_distance(color1, color2):
        return math.sqrt((color1[0] - color2[0]) ** 2 + (color1[1] - color2[1]) ** 2 + (color1[2] - color2[2]) ** 2)

    def total_distance(colors1, colors2):
        total_dist = 0
        for color1, color2 in zip(colors1, colors2):
            dist = euclidean_distance(color1, color2)
            total_dist += dist
        return total_dist

    three_list = []
    ans = []
    min_dist = None
    for k in range(3):
        fir = 0
        fir2 = 0
        for i, items in enumerate(new_mapping):
            three_list.append(items)
            fir += 1
            if fir == 3:
                fir = 0
                dist = total_distance(new_list_RGB[k], three_list)
                three_list = []
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    fir2 = i
        ans.append(fir2)

    fin = []
    for i in range(len(ans)):
        f = (ans[i]+1) / 9
        fin.append(emotion[int(f)])

    session['fin'] = fin
    save_emotions_to_database('static/uploads/156x200.jpg', fin,db_config)

    return graph_image_path, clustered_image_path



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)

        # 업로드된 이미지의 파일명을 세션에 저장
        session['uploaded_image'] = filename

        # 색상 추출 및 그래프 생성
        graph_image_path, clustered_image_path = extract_colors_and_graph(file_path)  # unpacking 필요
        session['color_graph'] = graph_image_path  # 색상 그래프 이미지 경로 저장

        return redirect(url_for('recomm'))


if __name__ == '__main__':
    app.run(debug=True, port=5001)
