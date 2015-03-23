# -*- coding: cp936 -*-
# 功能：抓取验证码
# 并存放到img目录下
# 文件名为图像的MD5
import urllib2
import hashlib
import time
# hack CERTIFICATE_VERIFY_FAILED
# https://github.com/mtschirs/quizduellapi/issues/2
import ssl
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

def download_img():
    pic_url = 'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand&0.21191171556711197'
    resp = urllib2.urlopen(pic_url)
    raw = resp.read()
    fn = hashlib.md5(raw).hexdigest()
    with open("img/%s.jpg" % fn, 'wb') as fp:
        fp.write(raw)

if __name__ == '__main__':
    i = 0
    while True:
        try:
#            time.sleep(1)  # 这里根本不需要等待
            download_img()
            i += 1
            print i
        except:
            print 'error'
