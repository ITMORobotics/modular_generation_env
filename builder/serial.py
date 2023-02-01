from __future__ import annotations
from itmobotics_sim.pybullet_env.urdf_editor import URDFEditor

from typing import Tuple

from spatialmath import SE3, SO3, Twist3
from spatialmath import base as sb

class SerialRobot:
    def __init__( self, 
        name: str,
        id: int,
        urdf_filename: str,
        connect_link: str,
        allow_connect: list,
        connect_tf: SE3 = SE3(),
    ):
        self.__name = name
        self.__id = id
        self.__urdf_filename = urdf_filename
        self.__urdf_editor = URDFEditor(self.__urdf_filename)
        self.__connect_tf = connect_tf
        self.__connect_link = connect_link

        self.__allow_connect = allow_connect
        self.__child_module = None
    
    def join_module(self, module: SerialRobot) -> bool:
        if not module in self.__allow_connect:
            return False
        self.__urdf_editor.joinURDF(module.urdf_editor, self.__connect_link, self.__connect_tf.A)
        self.__connect_link = module.connect_link
        self.__connect_tf = module.connect_tf
        self.__child_module = module
        return True

    def save(self, urdf_filename:str = None):
        if not urdf_filename:
            urdf_filename = self.__urdf_filename
        self.__urdf_editor.save(self.__urdf_filename)
    
    @property
    def chain(self) -> Tuple[str]:
        chain_list = [self.__name]
        child_module = self.__child_module
        while not child_module is None:
            chain_list.append(child_module.name)
            child_module = child_module.child_module
        return tuple(chain_list)
    
    @property
    def chain_id(self) -> Tuple[str]:
        chain_id_list = [self.__id]
        child_module = self.__child_module
        while not child_module is None:
            chain_id_list.append(child_module.id)
            child_module = child_module.child_module
        return tuple(chain_id_list)

    @property
    def urdf_filename(self) -> str:
        return self.__urdf_filename
    
    @property
    def urdf_editor(self) -> URDFEditor:
        return self.__urdf_editor
    
    @property
    def connect_tf(self) -> SE3:
        return self.__connect_tf
    
    @property
    def connect_link(self):
        return self.__connect_link
    
    @property
    def name(self):
        return self.__name
    
    @property
    def id(self) -> int:
        return self.__id
    
    @property
    def allow_connect(self):
        return self.__allow_connect
    
    @property
    def child_module(self):
        return self.__child_module
    
    @urdf_filename.setter
    def urdf_filename(self, new_filename: str):
        self.__urdf_filename = new_filename
