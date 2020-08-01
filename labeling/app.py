import pickle
import os

from flask import render_template
from flask import send_from_directory
from flask import request
from flask import redirect
from flask import Flask

app = Flask(__name__, static_url_path='')

done_img_url = 'https://i0.wp.com/operationrainfall.com/wp-content/uploads/2012/10/Gintama-End.jpg'

def get_img_urls_file_of_user(user_id):
  return os.path.join('labeling', 'users', user_id, 'unlabeled_img_urls.pkl')

def get_labels_file(user_id):
  return os.path.join('labeling', 'users', user_id, 'labels.csv')

def get_unlabeled_img_urls_of_user(user_id):
  img_urls_file = get_img_urls_file_of_user(user_id)
  with open(img_urls_file, 'rb') as f:
    unlabeled_img_urls = pickle.load(f)
  return unlabeled_img_urls

def mark_labeled(user_id, img_url):
  unlabeled_img_urls = get_unlabeled_img_urls_of_user(user_id)
  try:
    unlabeled_img_urls.remove(img_url)
    img_urls_file = get_img_urls_file_of_user(user_id)
    with open(img_urls_file, 'wb') as f:
      pickle.dump(unlabeled_img_urls, f, pickle.HIGHEST_PROTOCOL)
  except ValueError:
    pass

def get_unlabeled(user_id):
  unlabeled_img_urls = get_unlabeled_img_urls_of_user(user_id)
  if unlabeled_img_urls == []:
    img_url = done_img_url
  else:
    img_url = unlabeled_img_urls[-1]
  return img_url, len(unlabeled_img_urls)

def labeling_page(user_id, prev_label=None):
  img_url, n_unlabeled_imgs = get_unlabeled(user_id)
  buttons = [[str(i), str(i), 'checked' if str(i) == prev_label else ''] for i in range(10)]
  buttons.append(['idk', 'Not sure', 'checked' if 'idk' == prev_label else ''])
  return render_template('main.html', img_url=img_url, user_id=user_id, buttons=buttons, n_unlabeled_imgs=n_unlabeled_imgs)

@app.route('/imgs/<path:path>')
def send_img(path):
  return send_from_directory('imgs', path)

@app.route('/log_label', methods=['POST'])
def log_label():
  assert request.method == 'POST'
  assert 'submit' in request.form
  user_id = request.form['user_id']
  img_url = request.form['img_url']
  label = None
  if 'label' in request.form and img_url != done_img_url:
    label = request.form['label']
    labels_file = get_labels_file(user_id)
    with open(labels_file, 'a+') as f:
      f.write('%s,%s\n' % (img_url, label))
    mark_labeled(user_id, img_url)
  return labeling_page(user_id, prev_label=label)

@app.route('/', methods=['GET'])
def main():
  user_id = request.args['userid']
  return labeling_page(user_id)

if __name__ == '__main__':
  app.run(port=5000)
