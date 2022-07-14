import asyncio
import time
from io import BytesIO
from random import randint
from typing import Tuple

import httpx
import jieba
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from wordcloud import STOPWORDS, WordCloud

LIVELIVE: dict
FOLDER = 'live/'
Headers = {
    'Connection': 'keep-alive',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'
}


async def exface(session: httpx.AsyncClient()) -> Tuple[str, Image.Image]:
    resp = await session.get('https://account.bilibili.com/api/member/getCardByMid?mid=434334701', timeout=10.0)
    js = resp.json()
    face = js.get('card', {}).get('face')
    pendant = js.get('card', {}).get('pendant', {}).get('image')
    
    # 头像
    if isinstance(face, str):
        response = await session.get(face)  # 请求图片
        face = Image.open(BytesIO(response.read()))  # 读取图片
    w, h = face.size

    a = Image.new('L', face.size, 0)  # 创建一个黑色背景的画布
    ImageDraw.Draw(a).ellipse((0, 0, a.width, a.height), fill=255)  # 画白色圆形

    # 装扮
    if pendant:
        image = Image.new('RGBA', (int(1.75*w), int(1.75*h)), (0, 0, 0, 0))
        image.paste(face, (int(0.375*w), int(0.375*h)), mask=a)  # 粘贴至背景
        response = await session.get(pendant)  # 请求图片
        pd = Image.open(BytesIO(response.read()))  # 读取图片
        pd = pd.resize((int(1.75*w), int(1.75*h)), Image.ANTIALIAS)  # 装扮应当是头像的1.75倍
        try:
            image.paste(pd, (0, 0), mask=pd.getchannel('A'))  # 粘贴至背景
            return 'face', image
        except Exception:
            pendant = None
    # 粉圈
    if not pendant:
        image = Image.new('RGBA', (int(1.16*w), int(1.16*h)), (0, 0, 0, 0))
        image.paste(face, (int(0.08*w), int(0.08*h)), mask=a)  # 粘贴至背景
        ps = Image.new("RGB", (int(1.16*w), int(1.16*h)), (242, 93, 142))
        a = Image.new('L', ps.size, 0)  # 创建一个黑色背景的画布
        ImageDraw.Draw(a).ellipse((0, 0, a.width, a.height), fill=255)  # 画白色外圆
        ImageDraw.Draw(a).ellipse((int(0.06*w), int(0.06*h), int(1.1*w), int(1.1*h)), fill=0)  # 画黑色内圆
        image.paste(ps, (0, 0), mask=a)  # 粘贴至背景
        w, h = image.size
        bg = Image.new('RGBA', (int(1.25*w), int(1.25*h)), (0, 0, 0, 0))
        bg.paste(image, (int((1.25-1)/2*w), int((1.25-1)/2*h)))
        return 'face', bg


async def word2pic(client: httpx.AsyncClient()) -> Tuple[str, Image.Image]:
    resp = await client.get('https://api.drelf.cn/live/21452505/last', timeout=10.0)
    assert resp.status_code == 200
    global LIVEINFO 
    LIVEINFO = resp.json()
    response = await client.get(LIVEINFO['live']['cover'])  # 请求图片
    cover = Image.open(BytesIO(response.read()))  # 读取图片

    word = [u['msg'] for u in LIVEINFO['live']['danmaku'] if u['type'] == 'DANMU_MSG']
    word = jieba.cut('/'.join(word), cut_all=False)
    word = '/'.join(word)
    bg = Image.open(FOLDER+'shark.png')
    graph = np.array(bg)

    # 停用词
    content = [line.strip() for line in open(FOLDER+'stopwords.txt', 'r', encoding='utf-8').readlines()]
    stopwords = set(content) | STOPWORDS

    # 词云
    wc = WordCloud(
        font_path=FOLDER+'HarmonyOS_Sans_SC_Regular.ttf',
        prefer_horizontal=1,
        collocations=False,
        background_color=None,
        mask=graph,
        stopwords=stopwords,  
        mode="RGBA")
    wc.generate(word)
    return 'wc', [wc.to_image(), cover]


async def get_data(session: httpx.AsyncClient(), url: str, key: str) -> Tuple[str, dict]:
    r = await session.get(url)
    if r.status_code == 200:
        js = r.json()
        current_date = None
        count = 0
        pos = len(js)-1
        data_dict = {}
        while pos >= 0 and count < 5:
            date = time.strftime('%m-%d', time.localtime(js[pos]['time']/1000))
            if not current_date == date:
                try:
                    data_dict[date] = js[pos][key]
                except Exception:
                    pass
                count += 1
                current_date = date
            pos -= 1
        return key, data_dict
    else:
        print(key+'网络错误')


def circle_corner(img: Image.Image, radii: int = 0) -> Image.Image:  # 把原图片变成圆角，这个函数是从网上找的
    """
    圆角处理
    :param img: 源图象。
    :param radii: 半径，如：30。
    :return: 返回一个圆角处理后的图象。
    """
    if radii == 0:
        radii = int(0.1*img.height)
    else:
        radii = int(radii)

    # 画圆（用于分离4个角）
    circle = Image.new('L', (radii * 2, radii * 2), 0)  # 创建一个黑色背景的画布
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, radii * 2, radii * 2), fill=255)  # 画白色圆形

    # 画4个角（将整圆分离为4个部分）
    w, h = img.size
    alpha = Image.new('L', img.size, 255)
    alpha.paste(circle.crop((0, 0, radii, radii)), (0, 0))  # 左上角
    alpha.paste(circle.crop((radii, 0, radii * 2, radii)), (w - radii, 0))  # 右上角
    alpha.paste(circle.crop((radii, radii, radii * 2, radii * 2)), (w - radii, h - radii))  # 右下角
    alpha.paste(circle.crop((0, radii, radii, radii * 2)), (0, h - radii))  # 左下角
    
    img = img.convert('RGBA')
    img.putalpha(alpha)

    return img


async def makePic():
    bg = Image.open(FOLDER+'bg.png')
    draw = ImageDraw.Draw(bg)
    font = lambda size: ImageFont.truetype(FOLDER+'HarmonyOS_Sans_SC_Regular.ttf', size)
    fontbd = lambda size: ImageFont.truetype(FOLDER+'HarmonyOS_Sans_SC_Bold.ttf', size)
    t2s = lambda tt: time.strftime('%m/%d %H:%M:%S', time.localtime(tt))
    set_color = '#1D1D1F'

    basicData = [
        [],
        []
    ]

    async with httpx.AsyncClient(headers=Headers) as session:
        pending = [
            asyncio.create_task(word2pic(session)),
            asyncio.create_task(exface(session)),
            asyncio.create_task(get_data(session, 'https://api.tokyo.vtbs.moe/v2/bulkActiveSome/434334701', 'follower')),
            asyncio.create_task(get_data(session, 'https://api.tokyo.vtbs.moe/v2/bulkGuard/434334701', 'guardNum'))
        ]
        while pending:
            done, pending = await asyncio.wait(pending)
            for done_task in done:
                code, data = await done_task
                if code == 'wc':
                    wc = data[0].resize((900, 676), Image.ANTIALIAS)
                    bg.paste(wc, (90, 590), mask=wc.getchannel('A'))
                    cover = circle_corner(data[1]).resize((data[1].width*130//data[1].height, 130), Image.ANTIALIAS)
                    bg.paste(cover, (210, 1195), mask=cover.getchannel('A'))
                elif code == 'face':
                    face = data.resize((150, 150), Image.ANTIALIAS)
                    bg.paste(face, (55, 1185), mask=face.getchannel('A'))
                elif code == 'follower':
                    basicData[0].append(('关注：', ' → '.join([str(d) for d in data.values()][::-1])))
                elif code == 'guardNum':
                    basicData[1].append(('大航海：', ' → '.join([str(d) for d in data.values()][::-1])))

    draw.text((70, 178), '标题：'+LIVEINFO["live"]["title"], fill=set_color, font=fontbd(32))
    draw.text((70, 228), f'{t2s(LIVEINFO["live"]["st"])} - {t2s(LIVEINFO["live"]["sp"])}', fill=set_color, font=font(28))
    draw.text((240, 280), '关注 大航海为近五天数据', fill='grey', font=font(32))
    draw.text((240, 544), '出于个人爱好未将“妈妈”设为停止词', fill='grey', font=font(32))
    draw.text((600, 1290), '*数据来源：api.nana7mi.link', fill='grey', font=font(30))
    draw.text((140, 1440), '你们会无缘无故的发可爱，就代表哪天无缘无故发恶心', fill='grey', font=font(32))

    draw.text((70, 105), '直播记录', fill=set_color, font=fontbd(50))
    draw.text((70, 271), '基础数据', fill=set_color, font=fontbd(40))
    draw.text((70, 535), '弹幕词云', fill=set_color, font=fontbd(40))

    basicData = [
        [('弹幕：', LIVEINFO["live"]['total']), ('密度：', str(LIVEINFO["live"]['total']*60//(LIVEINFO["live"]['sp']-LIVEINFO["live"]['st']))+' / min')],
        [('礼物：', round(LIVEINFO["live"]['send_gift'], 2)), ('航海：', round(LIVEINFO["live"]['guard_buy'], 2)), ('醒目留言：', round(LIVEINFO["live"]['super_chat_message'], 2))],
    ] + basicData

    for i, rows in enumerate(basicData):
        for j, data in enumerate(rows):
            draw.text((70+240*j, 329+51*i), data[0], fill=set_color, font=font(35))
            draw.text((70+240*j+35*len(data[0]), 329+51*i+4), str(data[1]), fill=set_color, font=font(32))

    nanami = Image.open(f'{FOLDER}{randint(0,6)}.png')
    w = int(nanami.width*600/nanami.height)
    nanami = nanami.resize((w, 600), Image.ANTIALIAS)
    body = nanami.crop((0, 0, w, 400))  # 不是跟身体切割了吗

    a = body.getchannel('A')
    pix = a.load()
    for i in range(351, 400):
        for j in range(w):
            pix[j, i] = int((8-0.02*i) * pix[j, i])

    bg.paste(body, (885-w//2, 20), mask=a)

    card = Image.open(f'{FOLDER}card{randint(0,1)}.png')
    card = card.resize((100, 100), Image.ANTIALIAS)
    bg.paste(card, (20, 1400), mask=card.getchannel('A'))

    return bg


if __name__ == '__main__':
    bg = asyncio.run(makePic())
    bg.save('live.png')
