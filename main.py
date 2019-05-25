# Classifing handwritten digits with a toy Neural Network
# using the MNIST Database http://yann.lecun.com/exdb/mnist/

from mlxtend.data import loadlocal_mnist
from lib.nn import NeuralNetwork
from tqdm import tqdm
import numpy
import pygame


def train(brain, train_data, label_data):
    print("Training...")
    for i in tqdm(range(len(train_data))):
        inputs = train_data[i]
        targets = [0] * 10
        targets[label_data[i]] = 1
        brain.train(inputs, targets)
    brain.save()
    print("DONE!")


pygame.font.init()
prediction_font = pygame.font.SysFont('Comic Sans MS', 300)
correct_font = pygame.font.SysFont('Comic Sans MS', 100)
brain_font = pygame.font.SysFont('Arial', 30)

brain = NeuralNetwork(784, 16, 10)

size = width, height = 600, 300
screen = pygame.display.set_mode(size)

train_images, train_labels = loadlocal_mnist("data/train-images-idx3-ubyte",
                                             "data/train-labels-idx1-ubyte")

test_images, test_labels = loadlocal_mnist("data/t10k-images-idx3-ubyte",
                                           "data/t10k-labels-idx1-ubyte")

train_images_normed = train_images / 255
test_images_normed = test_images / 255

dumb_brain = NeuralNetwork(784, 16, 10)
info = numpy.load("trained_NN_info.npz")
smart_brain = NeuralNetwork.from_trained(info)
# train(dumb_brain, train_images, train_labels)
brain = dumb_brain
counter = 0
run = True
frame_count = 0
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                pygame.image.save(screen, "frame{}.png".format(frame_count))
            if event.key == pygame.K_n:
                counter += 1
            if event.key == pygame.K_p:
                counter -= 1
            if event.key == pygame.K_s:
                brain = smart_brain
            if event.key == pygame.K_d:
                brain = dumb_brain

    frame_count += 1
    screen.fill((255, 255, 255))
    test_index = counter % len(test_images_normed)
    if brain == dumb_brain:
        brain_text = "Dumb brain"
    elif brain == smart_brain:
        brain_text = "Smart brain"

    brain_surface = brain_font.render(brain_text, False, (0, 0, 0))
    screen.blit(brain_surface, (0.6*width, 0.05*height))
    inputs = test_images_normed[test_index]
    prediction = numpy.argmax(brain.predict(inputs))
    correct = test_labels[test_index]
    if prediction == correct:
        color = (10, 200, 10)
    else:
        color = (200, 10, 10)
        correct_surface = correct_font.render(
            str(correct), False, (10, 10, 200))
        screen.blit(correct_surface, (0.9*width, 0.8*height))

    prediction_surface = prediction_font.render(str(prediction), False, color)
    screen.blit(prediction_surface, (width/2+80, 70))
    pygame.draw.line(screen, (0, 0, 0), (width/2, 0), (width/2, height), 5)
    img = pygame.image.frombuffer(test_images[test_index], (28, 28), "P")
    img.set_colorkey((0, 0, 0))
    img = pygame.transform.scale(img, (int(width/2), height))
    screen.blit(img, (0, 0))
    pygame.display.update()
