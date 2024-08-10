from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# TODO: For every file in original proj
event_acc = event_accumulator.EventAccumulator("..\original_proj\logs\en_bert\events.out.tfevents.1721666458.orian-System-Product-Name.199637.9")
event_acc.Reload()

for scalar in ['train/loss', 'eval/loss']:
    print(scalar)
    x = []
    y = []
    for event in event_acc.Scalars(scalar):
        print(f"step: {event.step}, value: {event.value}")
        x.append(event.step)
        y.append(event.value)

    plt.plot(x, y)
    plt.show()
