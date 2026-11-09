from pathlib import Path
import shutil,plistlib
root=Path.cwd();src=root/'outputs/tap-model/live';dest=root/'outputs/TapTest.app'
s=(src/'TapPredictor.swift').read_text()
s=s.replace('if event.keyCode==49 { controller?.toggle() }','if event.keyCode==49 { controller?.toggle() }')
start=s.index('        else if event.keyCode==124')
end=s.index('        else {super.keyDown',start)
s=s[:start]+s[end:]
s=s.replace('override func mouseDown(with event:NSEvent){controller?.mark(convert(event.locationInWindow,from:nil))}','override func mouseDown(with event:NSEvent){}')
s=s.replace('else if c.prediction != nil {text="→ — следующий тап"}','else if c.prediction != nil {text="Коснись экрана ещё раз"}')
a=s.index('        let hint=');b=s.index('\n',a)
s=s[:a]+'        let hint="Пробел — начать / пауза · один тап — один прогноз · ⌘Q — выход"'+s[b:]
s=s.replace('if locked{text += confirmed ? " · подтверждено →" : " · ждёт отметку мышью"}','if locked{text += " · распознаю"}')
s=s.replace('self.status()\n                    }catch','self.locked=false\n                        self.status()\n                    }catch')
s=s.replace('window.title="Tap Predictor — области и клетка"','window.title="Tap Test — свободная проверка"')
assert 'self.locked=false\n                        self.status()' in s
out=src/'test-mode';out.mkdir(exist_ok=True);(out/'TapTest.swift').write_text(s)
(root/'work/main.swift').write_text(s)
if dest.exists():shutil.rmtree(dest)
shutil.copytree(root/'outputs/TapPredictor.app',dest,copy_function=shutil.copyfile)
p=dest/'Contents/Info.plist';info=plistlib.loads(p.read_bytes());info.update(CFBundleIdentifier='local.taptest',CFBundleDisplayName='Tap Test',CFBundleName='TapTest');p.write_bytes(plistlib.dumps(info))
w=(src/'worker.py').read_text();a=w.index('if manifest_path.exists():');b=w.index('bundle=joblib.load(active_path)',a);w=w[:a]+w[b:];(dest/'Contents/Resources/worker.py').write_text(w)
print('Prepared TapTest: automatic rearm, no mouse labels, base model forced')
