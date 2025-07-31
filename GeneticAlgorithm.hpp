#ifndef GENETICALGORITHM_HPP
#define GENETICALGORITHM_HPP

#include <vector>
#include <functional>

namespace afro
{
    
    enum class GADataType
    {
        Integer,
        Float,
        Double
    };
    
    struct GAValue
    {
        GADataType dataType = GADataType::Integer;
        union NumberValue
        {
            float f;
            double d;
            int i;
        };
        
        NumberValue numberValue;
        NumberValue minValue;
        NumberValue maxValue;
    };
    
   struct FitnessData
   {
      size_t index = 0;
      double fitness = -1;
   };
    
    class GeneticAlgorithm
    {
        public:
            GeneticAlgorithm(const std::vector<GAValue>& initialValues,
                                size_t populationSize,
                                size_t epochs,
                                size_t numParents,
                                size_t numGenesToModify,
                                void* userData,
                                std::function<double(GAValue*, size_t, void*)> fitnessFunction);
        
            virtual ~GeneticAlgorithm();
        
            bool advance();
        
            int getProgress() const;
        
            bool hasSolution() const;
        
            double getBestScore() const;
        
            const std::vector<GAValue>& getBestSolution();
        private:
            void update();
        
            void mutate(GAValue& value);
        
            void crossOver(GAValue& value1, GAValue& value2);
        
            void translate(GAValue& value);
        
            void ensureLimits(GAValue& value);
        
            void copyGenes(size_t parent, size_t child);
        
            void evaluateFitnessFunction();
        
            std::vector<GAValue> mSolutions;
            std::vector<FitnessData> mFitnessData;
            std::vector<GAValue> mBestSolution;
            double mBestSolutionIndex = 0;
            size_t mValueCount = 1;
            size_t mPopulation = 1;
            size_t mCurrentEpoch = 0;
            size_t mEpochs = 1;
            size_t mNumParents = 1;
            size_t mNumGenesToModify = 1;
            void* mUserData = nullptr;
            std::function<double(GAValue*, size_t, void*)> mFitnessFunction;
    };
    
}

#endif //GENETICALGORITHM_HPP
