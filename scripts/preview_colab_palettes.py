"""Render three safe public homepage previews and verify palette switching.

Requires Playwright and Chromium, like check_colab_study.py.
No notebook execution or model training.
"""
import json,hashlib,threading
from pathlib import Path
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from playwright.sync_api import sync_playwright,expect
root=Path(__file__).resolve().parents[1];out=root/'outputs/colab_study';target=root/'docs/previews';target.mkdir(exist_ok=True)
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(root)));threading.Thread(target=server.serve_forever,daemon=True).start();origin='http://127.0.0.1:'+str(server.server_port)
result={'palettes':{},'commands':{'cwd':str(root),'build':'python scripts/build_colab_study.py'},'browser_errors':[],'external_requests':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox']);result['browser_version']=browser.version
 for name in ['navy','olive','rose']:
  context=browser.new_context(viewport={'width':1440,'height':1220},reduced_motion='reduce');page=context.new_page()
  page.on('pageerror',lambda e:result['browser_errors'].append(str(e)))
  page.on('request',lambda r:result['external_requests'].append(r.url) if not r.url.startswith(origin+'/') else None)
  page.goto(origin+'/docs/belajar_colab_AI_interaktif.html?palette='+name+'&theme=light')
  assert page.locator('html').get_attribute('data-palette')==name
  assert page.locator('.palette-choice[aria-pressed=true]').count()==1
  assert page.locator('[data-palette-choice='+name+']').get_attribute('aria-pressed')=='true'
  page.screenshot(path=str(target/f'colab_classy_{name}.png'))
  colors=page.evaluate('({background:getComputedStyle(document.body).backgroundColor,hero:getComputedStyle(document.querySelector(".hero")).backgroundColor,accent:getComputedStyle(document.querySelector(".hero h1 em")).color})')
  for choice in ['navy','olive','rose']:
   page.locator('[data-palette-choice='+choice+']').click()
   assert page.locator('html').get_attribute('data-palette')==choice
  page.locator('[data-palette-choice='+name+']').focus();page.keyboard.press('Enter')
  assert page.evaluate('JSON.parse(localStorage.getItem("aes-study-v2:palette"))')==name
  page.goto(origin+'/docs/belajar_colab_AI_interaktif.html')
  assert page.locator('html').get_attribute('data-palette')==name
  for width in [390,768,1024,1440]:
   page.set_viewport_size({'width':width,'height':844})
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(name,width)
   page.locator('#tab-scenario').click()
   page.locator('#desk-tab-scenario-guided').click()
   assert page.locator('#page-scenario .margin-note').count()>=7
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(name,width,'code')
   page.locator('#tab-mulai').click()
  page.locator('#theme').click()
  assert page.locator('html').get_attribute('data-theme')=='dark'
  assert page.locator('html').get_attribute('data-palette')==name
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
  result['palettes'][name]={'colors':colors,'preview':f'docs/previews/colab_classy_{name}.png','png_sha256':hashlib.sha256((target/f'colab_classy_{name}.png').read_bytes()).hexdigest(),'widths_checked':[390,768,1024,1440],'keyboard_activation':True,'persisted_after_reload':True,'dark_theme_compatible':True}
  context.close()
 browser.close()
server.shutdown()
assert not result['browser_errors'] and not result['external_requests']
result['html_sha256']=hashlib.sha256((root/'docs/belajar_colab_AI_interaktif.html').read_bytes()).hexdigest()
(out/'classy_palettes_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
