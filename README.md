# Low-complexity Tracking of the Myocardium in 2D Echocardiography

Article (pre-print): [https://arxiv.org/abs/2503.10431](https://arxiv.org/abs/2503.10431)

This is the code repository for the MyoTracker architecture described in the article. MyoTracker is a drastically simplified version of Meta AI's CoTracker/CoTracker2 architecture proposed by Karaev et al. (https://co-tracker.github.io/), with the size reduced from 24M/45M parameters to under 0.32M through removal of various components. The reduced size is still sufficient for working with 2D echocardiography data, yet it offers greater speed and lower compute requirements.

## Architecture

MyoTracker architecture (as defined in ```myotracker/model/```):

![myotracker_diagram.jpg](https://github.com/artemcher/myotracker/blob/main/assets/myotracker_diagram.jpg)

## Examples

Visual examples with a MyoTracker model trained to track the right ventricular myocardium:

![echo_val_10.gif](https://github.com/artemcher/myotracker/blob/main/assets/echo_val_10.gif)

![result.gif](https://github.com/artemcher/myotracker/blob/main/assets/result.gif)


## License

MyoTracker propagates CoTracker's license (CC-BY-NC 4.0).

## Citing

If you use the code in further work, it would be best to cite the original publication on CoTracker as well:

```bibtex
@article{karaev2023cotracker,
  title={CoTracker: It is Better to Track Together},
  author={Nikita Karaev and Ignacio Rocco and Benjamin Graham and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  journal={arXiv:2307.07635},
  year={2023}
}
```
