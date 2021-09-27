import numpy as np

def compute_dataset_normalization(dataloader, no_last_dim_norm=True):
    states_list = []
    action_list = []

    print('computing normalization....')
    for i_batch, sample_batched in enumerate(dataloader):
        if 'actions' not in sample_batched:
            raise NotImplementedError('todo!')
        states_list.append(sample_batched['states'])
        action_list.append(sample_batched['actions'])
        if i_batch == 200:
            break

        states = np.concatenate(states_list, axis=0)
        actions = np.concatenate(action_list, axis=0)
        if actions.shape[0] > 1000:
            break

    print('state dim: ', states.shape)
    print('action dim: ', actions.shape)
    if actions.shape[0] < 1000:
        print('Warning Very few examples found!!!')
        import pdb; pdb.set_trace()

    dict = {
    'states_mean' : np.mean(states, axis=0),
    'states_std' : np.std(states, axis=0),
    'actions_mean': np.mean(actions, axis=0),
    'actions_std': np.std(actions, axis=0),
    }

    for dim in range(states.shape[1]):
        if dict['states_mean'][dim] == 0 and dict['states_std'][dim] == 0:
            dict['states_mean'][dim] = 0
            dict['states_std'][dim] = 1
            print('##################################')
            print('not normalizing state dim {}, since mean and std are zero!!'.format(dim))
            print('##################################')

    for dim in range(actions.shape[1]):
        if dict['actions_mean'][dim] == 0 and dict['actions_std'][dim] == 0:
            dict['actions_mean'][dim] = 0
            dict['actions_std'][dim] = 1
            print('##################################')
            print('not normalizing action dim {}, since mean and std are zero!!'.format(dim))
            print('##################################')

    if no_last_dim_norm:
        print('##################################')
        print('not normalizing grasp action!')
        print('##################################')
        dict['actions_mean'][-1] = 0
        dict['actions_std'][-1] = 1

    print('normalization params')
    print(dict)

    return dict