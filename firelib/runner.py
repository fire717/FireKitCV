import time
import datetime
import gc
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from firelib.common_utils import print_dash
from firelib.runner_utils import get_loss, get_optimizer, get_scheduler, clip_gradient, GradualWarmupScheduler
from firelib.metrics import FireMetrics




class FireRunner():
    def __init__(self, cfg, model):

        self.cfg = cfg

  
        if self.cfg['GPU_ID'] != '' :
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        self.scaler = torch.cuda.amp.GradScaler()
        ############################################################
        

        self.loss_func = get_loss(self.device, cfg)

        self.optimizer = get_optimizer(self.cfg['optimizer'], 
                                    self.model, 
                                    self.cfg['learning_rate'], 
                                    self.cfg['weight_decay'])

        self.scheduler = get_scheduler(self.cfg['scheduler'], self.optimizer)

        self.train_metrics = FireMetrics()
        self.val_metrics = FireMetrics()
        
        if self.cfg['warmup_epoch']:
            self.scheduler = GradualWarmupScheduler(optimizer, 
                                                multiplier=1, 
                                                total_epoch=self.cfg['warmup_epoch'], 
                                                after_scheduler=self.scheduler)

    def train(self, train_loader, val_loader):

        self._on_train_start()

        for epoch in range(self.cfg['epochs']):

            self._freeze_layers(epoch)

            self._on_train_step(train_loader, epoch)

            self._on_validation(val_loader, epoch)

            if self.check_earlystop:
                break
        
        self._on_train_end()

    def evaluate(self, data_loader):
        self.model.eval()
        val_loss = 0
        correct = 0

        with torch.no_grad():
            pres = []
            labels = []
            for (data, target, img_names) in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                #print(target.shape)
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    val_loss += self.loss_func(output[0], target).item() # sum up batch loss

                pred_score = nn.Softmax(dim=1)(output[0])
                self.val_metrics.update(pred_score.detach().cpu().numpy().tolist(), target.detach().cpu().numpy().tolist())
 

        val_loss /= len(data_loader.dataset)
        val_acc =  self.val_metrics.get_acc()
        val_F1 =  self.val_metrics.get_F1()

        self.best_score = val_acc

        print(' \n           [VAL] loss: {:.4f}, acc: {:.4f}   F1: {:.4f}\n'.format(
                val_loss, val_acc, val_F1))



    def predict(self, data_loader, return_raw=False):
        self.model.eval()
        correct = 0

        res_dict = {}
        with torch.no_grad():
            # pres = []
            # labels = []
            for (data, img_names) in data_loader:
                data = data.to(self.device)

                output = self.model(data)[0]


                #print(output.shape)
                pred_score = nn.Softmax(dim=1)(output)
                #print(pred_score.shape)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

                batch_pred_score = pred_score.data.cpu().numpy().tolist()
                for i in range(len(batch_pred_score)):
                    if return_raw:
                        res_dict[os.path.basename(img_names[i])] = pred_score[i].cpu().numpy()
                    else:
                        res_dict[os.path.basename(img_names[i])] = pred[i].item()
        return res_dict

    def clean_data(self, data_loader, target_label, move_dir):
        """
        input: data, move_path
        output: None

        """
        self.model.eval()
        
        #predict
        #res_list = []
        count = 0
        with torch.no_grad():
            #end = time.time()
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r",str(i)+"/"+str(data_loader.__len__()),end="",flush=True)

                inputs = inputs.cuda()

                output = self.model(inputs)
                output = nn.Softmax(dim=1)(output)
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):

                    output_one = output[i][np.newaxis, :]

                    if np.argmax(output_one)!=target_label:
                        print(output_one, target_label,img_names[i])
                        img_name = os.path.basename(img_names[i])
                        os.rename(img_names[i], os.path.join(move_dir,img_name))
                        count += 1
        print("[INFO] Total: ",count)

    def load_model(self,model_path, data_parallel=False):
        self.model.load_state_dict(torch.load(model_path), strict=True)
        
        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def save_model(self, save_name):
        torch.save(self.model.state_dict(), save_name)

    def convert_onnx(self, save_name= "model.onnx"):
        dummy_input = torch.randn(1, 3, self.cfg['img_size'][0], self.cfg['img_size'][1]).to(self.device)

        torch.onnx.export(self.model, 
                        dummy_input, 
                        os.path.join(self.cfg['save_dir'],save_name), 
                        verbose=True)


############################################################
    def _make_save_dir(self):
        #exist_names = os.listdir(self.cfg['save_dir'])
        #print(os.walk(self.cfg['save_dir']))
        dirpath, dirnames, filenames = os.walk(self.cfg['save_dir']).__next__()
        exp_nums = []
        for name in dirnames:
            if name[:3]=='exp':
                try:
                    expid = int(name[3:])
                    exp_nums.append(expid)
                except:
                    continue
        new_id = 0
        if len(exp_nums)>0:
            new_id = max(exp_nums)+1
        exp_dir = os.path.join(self.cfg['save_dir'], 'exp'+str(new_id))

        print("Save to %s" % exp_dir)
        #if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

        return exp_dir

    def _freeze_layers(self, epoch, freeze_epochs=0, layers_not_freeze=-1):
        
        if epoch<freeze_epochs:
            print(f"Freezing layers:{layers_not_freeze} for epoch: {epoch}")
            for child in list(self.model.backbone.children())[:layers_not_freeze]:
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for child in list(self.model.backbone.children()):
                for param in child.parameters():
                    param.requires_grad = True

    def _on_train_start(self):
        
        self.last_best_value = 0
        self.last_best_dist = 0
        self.last_save_path = None

        self.check_earlystop = False
        self.best_epoch = 0

        # log
        self.log_time = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        print(f"Start time: {self.log_time}")

        self.exp_dir = self._make_save_dir()

    def _on_train_step(self,train_loader, epoch):
        
        self.model.train()
        correct = 0
        count = 0
        batch_time = 0
        total_loss = 0

        self.train_metrics.reset()
        time_start = time.time()

        for batch_idx, (data, target, img_names) in enumerate(train_loader):

            target = target.to(self.device)

            data = data.to(self.device)

            with torch.cuda.amp.autocast():
                output = self.model(data)
                #all_linear2_params = torch.cat([x.view(-1) for x in model.model_feature._fc.parameters()])
                #l2_regularization = 0.0003 * torch.norm(all_linear2_params, 2)
                loss = self.loss_func(output[0], target)# + l2_regularization.item()    


            total_loss += loss.item()
            if self.cfg['clip_gradient']:
                clip_gradient(self.optimizer, self.cfg['clip_gradient'])


            
            self.optimizer.zero_grad()#把梯度置零
            # loss.backward() #计算梯度
            # self.optimizer.step() #更新参数
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            ### train acc
            pred_score = nn.Softmax(dim=1)(output[0])
            self.train_metrics.update(pred_score.detach().cpu().numpy().tolist(), target.detach().cpu().numpy().tolist())
 
            count += len(data)
            train_loss = total_loss/count


            eta = int((time.time() - time_start)/(batch_idx+1)*(len(train_loader)-batch_idx-1))
            train_acc = self.train_metrics.get_acc()

            print_epoch = ''.join([' ']*(4-len(str(epoch+1))))+str(epoch+1)
            print_epoch_total = str(self.cfg['epochs'])+''.join([' ']*(4-len(str(self.cfg['epochs']))))

            log_interval = 10
            if batch_idx % log_interval== 0:
                print('\r',
                    '{}/{} [{}/{} ({:.0f}%)] - ETA: {}, loss: {:.4f}, acc: {:.4f}  LR: {}'.format(
                    print_epoch, print_epoch_total, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    datetime.timedelta(seconds=eta),
                    train_loss,train_acc,
                    str(self.optimizer.param_groups[0]["lr"])[:8]), 
                    end="",flush=True)

    def _on_train_end(self):
        save_name = 'last.pt'
        self.last_save_path = os.path.join(self.exp_dir, save_name)
        self.modelSave(self.last_save_path)
        
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg["cfg_verbose"]:
            printDash()
            print(self.cfg)
            printDash()

    def _on_validation(self, val_loader, epoch):
      
        self.evaluate(val_loader)

        if self.cfg['warmup_epoch']:
            self.scheduler.step(epoch)
        else:
            if 'default' in self.cfg['scheduler']:
                self.scheduler.step(self.best_score)
            else:
                self.scheduler.step()

        self._checkpoint(epoch)
        self._early_stop(epoch)

    def _on_test(self):
        self.model.eval()
 
        res_list = []
        with torch.no_grad():
            #end = time.time()
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)

                inputs = inputs.cuda()

                output = model(inputs)
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):

                    output_one = output[i][np.newaxis, :]
                    output_one = np.argmax(output_one)

                    res_list.append(output_one)
        return res_list

    def _early_stop(self, epoch):
        if self.best_score>self.last_best_value:
            self.last_best_value = self.best_score
            self.last_best_dist = 0

        self.last_best_dist+=1
        if self.last_best_dist>self.cfg['early_stop_patient']:
            self.best_epoch = epoch-self.cfg['early_stop_patient']+1
            print("[INFO] Early Stop with patient %d , best is Epoch - %d :%f" % (self.cfg['early_stop_patient'],self.best_epoch,self.last_best_value))
            self.check_earlystop = True
        if  epoch+1==self.cfg['epochs']:
            self.best_epoch = epoch-self.last_best_dist+2
            print("[INFO] Finish trainging , best is Epoch - %d :%f" % (self.best_epoch,self.last_best_value))
            self.check_earlystop = True

    def _checkpoint(self, epoch):
        
        if self.best_score<=self.last_best_value:
            pass
        else:
            save_name = 'best.pt'
            self.last_save_path = os.path.join(self.exp_dir, save_name)
            self.save_model(self.last_save_path)
