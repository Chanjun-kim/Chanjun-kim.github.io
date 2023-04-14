class DataWorker :
    """
    데이터직군에 대한 클래스입니다.
    """
    
    # 인스턴스를 만들 때 자동으로 실행됨
    def __init__(self, name : str) :
        """
        데이터 직군에 대한 정보를 초기화하는 함수입니다.

        Args:
            name (str): 이름
        """
        self.name = name
    
    # 인스턴스를 호출할 때마다 자동으로 실행됨
    def __call__(self) :
        """
        객체를 부를 때 대답합니다.
        """
        print('부르셨나요?')
    
    # 메소드
    def introduce(self) :
        """
        자기 소개를 하고 명함을 건내줍니다.

        Returns:
            _type_: 명함
        """
        print(f'안녕하세요 데이터 직군에 종사하는 {self.name} 입니다.')
    
        return f"명함 : {self.name}"
    

class DataEngineer(DataWorker) :
    """
    데이터 엔지니어에 대한 클래스입니다.
    """
    def __init__(self, name : str, working_list : list) :
        """
        데이터 엔지니어에 대한 정보를 초기화하는 함수입니다.

        Args:
            name (str): 이름
            working (list): 하는 업무 리스트
            _today_objective(str) : 오늘의 목표
        """
        super().__init__(name)
        self.working_list = working_list
        self._today_objective = "칼퇴"
    
    def __call__(self) :
        """
        데이터엔지니어를 부를 때 대답합니다.
        """
        super().__call__()
        print('그렇게 막 가져가시면 안되는데요.')
    
    def introduce(self) :
        print(f"데이터엔지니어 {self.name} 입니다.")
        for w in self.working_list :
            print(f"\t하는 업무는 {w} 입니다.")
    
    @property
    def today_objective(self) :
        return self._today_objective
    
    @today_objective.setter
    def today_objective(self, value) :
        if value in ["칼퇴", "야근"] :
            self._today_objective = value
        else : 
            print("칼퇴 아니면 야근 밖에 선택권이 없습니다.")
            raise ValueError
            


class DataScientist(DataWorker) :
    """
    데이터사이언티스트에 대한 클래스입니다.
    """
    def __init__(self, name : str, working : list, __job_turnover : bool) :
        """
        데이터 사이언티스트 대한 정보를 초기화하는 함수입니다.

        Args:
            name (str): 이름
            working (list): 하는 업무 리스트
            job_turnover(bool) : 공개한 이직 희망 의사
            __job_turnover(bool) : 솔직한 이직 희망 의사
        """
        super().__init__(name)
        self.working = working
        self.__job_turnover = __job_turnover
        self.job_turnover = False 
    
    def __call__(self) :
        """
        데이터사이언티스트를 부를 때 대답합니다.
        """
        super().__call__()
        print('그렇게 해석하시면 안되는데요')
    
    def introduce(self) :
        """
        자기소개를 합니다.
        """
        print(f"데이터사이언티스트 {self.name} 입니다.")
        for w in self.working :
            print(f"\t하는 업무는 {w} 입니다.")
    
    def question_turnover(self) :
        """
        이직 여부를 물어봅니다.
        """
        if self.job_turnover :
            print("다음주에 면접이에요")
        else :
            print("평생 다녀야죠!")
    
    def _drink_alchol(self) :
        """
        술을 마십니다.
        """
        self.job_turnover = self.__job_turnover
    


class Leader :
    
    def __init__(self, team_name) :
        self.team_name = team_name
    
    def leading(self):
        print(f"{self.team_name}팀 주간회의 시작하겠습니다.")
    

class _DataLeader(DataWorker, Leader) :
    """
    팀장에 대한 클래스입니다.
    """
    def __init__(self, name, team_name) :
        """
        팀장 정보를 초기화하는 함수입니다.

        Args:
            name (str): 이름
            team_name(str) : 팀 이름
        """
        DataWorker.__init__(self, name)
        Leader.__init__(self, team_name)



class __Spy(DataWorker) :
    """
    내부 스파이에 대한 클래스입니다.
    """
    def __init__(self, name, team_name) :
        """
        팀장 정보를 초기화하는 함수입니다.

        Args:
            name (str): 이름
        """
        super().__init__(name)
        self.team_name = team_name
        self.__is_spy = True
        self.is_spy = False
    
    def __call__(self) :
        """
        자기소개를 합니다.
        """        
        print(f"{self.team_name} 팀장 {self.name} 입니다.")
    
    @property
    def is_spy(self) :
        return self.is_spy
    
    @is_spy.setter 
    def is_spy(self, value : bool) :
        if type(value) == bool : 
            self.is_spy = value
        else :
            raise ValueError
    
    def must_say_truth_game(self) :
        """
        진실 게임을 합니다.
        진짜 정체를 밝힙니다.
        """
        self.is_spy = self.__is_spy
    
    def who_is_spy(self) :
        """
        누가 스파이인지 물어봅니다.
        """
        if is_spy == False :
            print("저는 스파이가 누군지 몰라요")
        else :
            print("스파이는 저였습니다.")

