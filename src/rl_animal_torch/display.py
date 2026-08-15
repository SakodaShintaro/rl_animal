import cv2

WINDOW = "animalai"
SCALE = 6


def show(visual, title):
    """
    The newest of the stacked frames, enlarged. Only used when a run has one environment,
    so nothing else is competing for the window.
    """
    frame = cv2.cvtColor(visual[:, :, -3:], cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    cv2.putText(frame, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow(WINDOW, frame)
    cv2.waitKey(1)


def close():
    cv2.destroyAllWindows()
