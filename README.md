# Python-Server

Model의 Serving을 담당해주는 Server입니다.

## Usage

```bash
flask run
```

## Model 추가 하는 법

구현하고자 하는 ML 서비스를 모듈별로 분리시켜서 구현을 해야 합니다.

- `train.py` : 학습을 담당하는 모듈
- `network.py` : 네트워크가 구현된 모듈
- `predict.py` : 추론을 담당하는 모듈
- `config.py` : 하이퍼파라미터를 설정할 수 있는 모듈

> 꼭 이러한 파일 구조를 따르지 않아도 됩니다.

중요한 점은 `train.py` 실행 후 model의 weight이 저장되어야 하고 `predict.py` 실행 시 model의 weight을 load후에 추론을 실행해야 합니다.

## Model 추가 상세 설명

`train.py`, `predict.py`, `app.py` 에서 어떤 방식으로 동작이 되어야 하는지 설명하겠습니다.

### train.py

[train.py 예제 코드](./imageClassification/train.py)

```python
def run(model, criterion, optimizer, train_dataloader, test_dataloader):
    best_valid_loss = float("inf")

    for epoch in range(cfg.EPOCHS):
        start_time = time.time()

        train_loss = train(model, criterion, optimizer, train_dataloader)
        valid_loss = evaluate(model, criterion, optimizer, test_dataloader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), cfg.MODEL_PATH)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
```

코드를 보시면 `cfg.MODEL_PATH` 경로에 model의 weight을 저장하는 것을 확인 하실 수 있습니다.

### predict.py

[predict.py 예제 코드](./imageClassification/predict.py)

```python
if not os.path.exists(cfg.MODEL_PATH):
    main()  # weight이 저장되지 않을 경우 train.main 실행

model = ImageClassiicationModel().to(cfg.DEVICE)
model.load_state_dict(torch.load(cfg.MODEL_PATH))
```

`cfg.MODEL_PATH` 경로에 파일이 존재할 경우 학습을 진행하지 않습니다. 만약 파일이 없을 경우 학습을 진행합니다.


```python
def image_predict(image):
    class_names = ["cat", "dog", "squirrel"]
    _, transform = get_transforms()
    image = transform(image).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        output = model(image)

        # torch.max(output, 1)
        pred = torch.argmax(output, dim=1)

    class_idx = pred.item()
    return class_names[class_idx]
```

flask server가 실행되면 model은 메모리 공간 상에 위치하게 됩니다.
`/imageCls` 경로로 요청이 들어올 경우 `image_predict` 함수가 실행됩니다.

### app.py

[app.py 예제 코드](app.py)

```python
from imageClassification.predict import image_predict

@app.route("/imageCls", methods=["POST"])
def imagefunc():
    content = request.get_json(force=True, silent=True)

    if request.method != "POST":
        return "통신 오류!!"

    try:
        img = content["imageFile"]
        img = base64.b64decode(img)
        buf = io.BytesIO(img)
        img = Image.open(buf)

        class_name = image_predict(img)

        return jsonify({"class_name": class_name})

    except KeyError:
        return "이미지를 넣어주세요!!!"
```

`from imageClassification.predict import image_predict` 라인이 실행되면서 모델의 `weight`이 저장되어있는지 체크를 한 후 학습이 진행된다고 이해하면 됩니다. (위에서 설명한 것)

> 여기까지 구현하셨으면 Front 부분은 멋진 개발자님이 척척 구현해주실 겁니다.
